#!/usr/bin/env python3
"""Run target-only (naive) PolarMACE fine tuning from prepared extxyz data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from ase.data import atomic_numbers

# e3nn 0.4.x stores trusted, package-shipped Wigner constants as a Torch
# archive.  PyTorch >= 2.6 defaults to weights_only=True and otherwise rejects
# the archive before MACE starts.  Set this before importing MACE/e3nn.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# e3nn 0.4.4 is pinned by MACE and dynamically generates FX modules that
# PyTorch 2.11 warns about while compiling them to TorchScript.  The generated
# modules are valid; suppress only this known compatibility warning.
warnings.filterwarnings(
    "ignore",
    message=(
        "The TorchScript type system doesn't support instance-level annotations "
        "on empty non-base types in `__init__`.*"
    ),
    category=UserWarning,
)

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
        f"--model={'MACE' if args.delta_correction else 'PolarMACE'}",
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
        f"--patience={args.patience}",
        f"--report_train_metrics={args.report_train_metrics}",
        f"--report_train_metrics_interval={args.report_train_metrics_interval}",
        f"--lr={args.lr}",
        f"--swa_lr={args.swa_lr}",
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
    argv.extend([
        f"--foundation_model={args.foundation_model}",
        "--multiheads_finetuning=False",
    ])
    if args.delta_correction:
        argv.append("--delta_correction=True")
    if args.restart_latest:
        argv.append("--restart_latest")
    if args.ema:
        argv.extend(["--ema", f"--ema_decay={args.ema_decay}"])
    if not args.no_swa:
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
    parser.add_argument("--name", default="polar1s_delta_154")
    parser.add_argument("--foundation-model", default="polar-1-s")
    parser.add_argument("--train-file", type=Path, default=DATA_DIR / "target_train_154_delta.xyz")
    parser.add_argument("--valid-file", type=Path, default=DATA_DIR / "target_valid_delta.xyz")
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR / "polar1s_delta_154")
    parser.add_argument(
        "--e0s",
        default="foundation",
        help="'foundation' (default, embedded MACE-POLAR E0s), 'estimated', 'dft', or an E0 JSON path.",
    )
    parser.add_argument(
        "--delta-correction", action="store_true", default=True,
        help=("Freeze Polar's representation and train a zero-initialized correction readout on "
              "DFT-minus-foundation labels. Its output must be added to the frozen foundation."),
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
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Stop after this many non-improving validation-loss epochs (default: 20).",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=250,
        help="Accepted for backward compatibility; current MACE controls progress logging internally.",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--default-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="cuda")
    # The default is a standalone residual model, not a full foundation update,
    # so it needs a conventional train-from-scratch learning rate.
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Adam learning rate for the small zero-initialized delta readout (default: 3e-4).",
    )
    parser.add_argument(
        "--swa-lr",
        type=float,
        default=3e-4,
        help="Stage Two learning rate (default: 3e-4, matching --lr).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--restart-latest", action="store_true")
    parser.add_argument(
        "--ema", action="store_true",
        help="Enable EMA averaging (off by default for the small delta-learning run).",
    )
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--start-swa", type=int, default=15)
    parser.add_argument("--swa-energy-weight", type=float, default=1.0)
    parser.add_argument("--swa-forces-weight", type=float, default=100_000.0)
    parser.add_argument("--no-swa", action="store_true", default=True, help="Disable MACE Stage Two/SWA.")
    parser.add_argument(
        "--report-train-metrics", action="store_true", default=True,
        help="Print full-train-set RMSE alongside validation RMSE periodically.",
    )
    parser.add_argument(
        "--report-train-metrics-interval", type=int, default=5,
        help="Epoch interval for full-train-set RMSE reporting (default: 5).",
    )
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
