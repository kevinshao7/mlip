#!/usr/bin/env python3
"""Run target-only (naive) PolarMACE fine tuning from prepared extxyz data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ase.data import atomic_numbers

# e3nn 0.4.x stores trusted, package-shipped Wigner constants as a Torch
# archive.  PyTorch >= 2.6 defaults to weights_only=True and otherwise rejects
# the archive before MACE starts.  Set this before importing MACE/e3nn.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:  # pragma: no cover - torch is required to train MACE.
    def record(function):
        return function


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
MACE_REPO = MLIP_DIR / "mace"
CACHE_DIR = MLIP_DIR / "outputsfull" / ".cache"
DATA_DIR = SCRIPT_DIR / "data"
RUNS_DIR = SCRIPT_DIR / "runs"
C2_ATOMIZATION_E0S = (
    SCRIPT_DIR.parent / "C2_atomizationDFT"
).parent / "7_7b_clustervalidation" / "atomizationenergies.txt"
DFT_E0S_JSON = DATA_DIR / "target_dft_e0s.json"


def path_arg(path: Path) -> str:
    return str(path.resolve())


def require_file(path: Path, message: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{message}: {path}")
    return path_arg(path)


def write_dft_e0_json(source: Path, destination: Path) -> str:
    """Convert the isolated-atom DFT CSV to MACE's atomic-number JSON format."""
    require_file(source, "Missing C2 DFT atomization-energy CSV")
    e0s: dict[int, float] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("atom"):
            continue
        symbol, value = (field.strip() for field in line.split(",", maxsplit=1))
        e0s[atomic_numbers[symbol]] = float(value)
    if not e0s:
        raise ValueError(f"No DFT E0 values found in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({str(z): value for z, value in sorted(e0s.items())}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Using DFT E0s from {source}: {e0s}")
    return path_arg(destination)


def resolve_e0s(e0s_arg: str) -> str:
    if e0s_arg.lower() == "dft":
        # Rewriting this file at launch is unnecessary after prepare_data.py
        # (or the bundled reference file) has created it.
        if DFT_E0S_JSON.is_file():
            print(f"Using prepared DFT E0s from {DFT_E0S_JSON}")
            return require_file(DFT_E0S_JSON, "Missing prepared DFT E0 JSON")
        return write_dft_e0_json(C2_ATOMIZATION_E0S, DFT_E0S_JSON)
    if e0s_arg.lower() == "foundation":
        return "foundation"
    if e0s_arg.lower() == "estimated":
        return "estimated"
    return require_file(Path(e0s_arg), "Missing E0 JSON")


def build_train_argv(args: argparse.Namespace) -> list[str]:
    train_file = require_file(args.train_file, "Missing target training extxyz")
    valid_file = require_file(args.valid_file, "Missing target validation extxyz")
    e0s = resolve_e0s(args.e0s)
    run_dir = args.run_dir.resolve()
    paths = {
        "model": run_dir / "models",
        "checkpoints": run_dir / "checkpoints",
        "log": run_dir / "logs",
        "results": run_dir / "results",
        "work": run_dir / "work",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    argv = [
        "run_train.py",
        f"--name={args.name}",
        "--model=PolarMACE",
        f"--foundation_model={args.foundation_model}",
        "--multiheads_finetuning=False",
        f"--train_file={train_file}",
        f"--valid_file={valid_file}",
        f"--atomic_numbers={args.atomic_numbers}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--total_charge_key=charge",
        "--total_spin_key=spin",
        f"--E0s={e0s}",
        f"--energy_weight={args.energy_weight}",
        f"--forces_weight={args.forces_weight}",
        "--stress_weight=0.0",
        "--loss=weighted",
        f"--batch_size={args.batch_size}",
        f"--valid_batch_size={args.valid_batch_size}",
        f"--max_num_epochs={args.max_num_epochs}",
        f"--lr={args.lr}",
        f"--log_interval={args.status_every}",
        f"--seed={args.seed}",
        f"--default_dtype={args.default_dtype}",
        f"--device={args.device}",
        f"--num_workers={args.num_workers}",
        f"--model_dir={path_arg(paths['model'])}",
        f"--checkpoints_dir={path_arg(paths['checkpoints'])}",
        f"--log_dir={path_arg(paths['log'])}",
        f"--results_dir={path_arg(paths['results'])}",
        f"--work_dir={path_arg(paths['work'])}",
    ]
    if args.restart_latest:
        argv.append("--restart_latest")
    if args.ema:
        argv.extend(["--ema", f"--ema_decay={args.ema_decay}"])
    # MACE's SWA ("Stage Two") also changes the loss weighting and lowers the
    # learning rate for the final training phase.  Keep it mandatory so every
    # fine-tuning run has the same convergence behavior.
    argv.extend([
        "--swa",
        f"--start_swa={args.start_swa}",
        f"--swa_energy_weight={args.swa_energy_weight}",
        f"--swa_forces_weight={args.swa_forces_weight}",
    ])
    if args.dry_run:
        print(" ".join(argv))
        raise SystemExit(0)
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="polar1s_naive_orca_dft_e0")
    parser.add_argument("--foundation-model", default="polar-1-s")
    parser.add_argument("--train-file", type=Path, default=DATA_DIR / "target_train.xyz")
    parser.add_argument("--valid-file", type=Path, default=DATA_DIR / "target_valid.xyz")
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR / "polar1s_naive_orca_dft_e0")
    parser.add_argument(
        "--e0s",
        default="foundation",
        help="'foundation' (default, embedded MACE-POLAR E0s), 'estimated', 'dft', or an E0 JSON path.",
    )
    parser.add_argument("--atomic-numbers", default="[1, 7, 8, 16]")
    # Preserve the established force-loss scale while making forces dominate
    # energy by 100,000:1. Raising forces_weight instead would rescale the
    # whole loss and interact badly with the fixed learning rate/clipping.
    parser.add_argument("--energy-weight", type=float, default=0.001)
    parser.add_argument("--forces-weight", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--valid-batch-size", type=int, default=1)
    parser.add_argument("--max-num-epochs", type=int, default=200)
    parser.add_argument("--status-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--default-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="cuda")
    # Fine tuning begins from an already accurate foundation checkpoint.  The
    # MACE base default (1e-2) is a pre-training-scale step and caused the
    # validation error to jump after the first batch-size-one epoch.
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--restart-latest", action="store_true")
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.99999)
    parser.add_argument("--start-swa", type=int, default=15)
    parser.add_argument("--swa-energy-weight", type=float, default=1.0)
    parser.add_argument("--swa-forces-weight", type=float, default=100_000.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


@record
def main() -> None:
    args = parse_args()
    if not MACE_REPO.is_dir():
        raise FileNotFoundError(f"Local MACE checkout is missing: {MACE_REPO}")
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))
    sys.path.insert(0, str(MACE_REPO.resolve()))
    sys.argv = build_train_argv(args)
    from mace.cli.run_train import main as run_train_main
    run_train_main()


if __name__ == "__main__":
    main()
