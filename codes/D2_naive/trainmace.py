#!/usr/bin/env python3
"""Run target-only (naive) PolarMACE fine tuning from prepared extxyz data."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
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


def unique_run_context(args: argparse.Namespace) -> tuple[str, Path]:
    """Return an immutable, timestamped directory for one training attempt."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.name).strip(".-")
    if not safe_name:
        raise ValueError("--name must contain at least one letter or number")
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    run_name = f"{safe_name}_{timestamp}"
    run_dir = args.runs_root.resolve() / run_name
    if run_dir.exists():  # Microseconds make this exceptional, but never overwrite a run.
        raise FileExistsError(f"Refusing to reuse an existing run directory: {run_dir}")
    return run_name, run_dir


def append_experiment_log(args: argparse.Namespace, run_name: str, run_dir: Path, argv: list[str]) -> None:
    """Append one launch record to the workflow's single experiment log."""
    destination = SCRIPT_DIR / "hyperparameters.txt"
    lines = [
        "",
        "=" * 88,
        f"RUN START {datetime.now().astimezone().isoformat()}",
        f"created_at={datetime.now().astimezone().isoformat()}",
        f"run_name={run_name}",
        f"run_directory={run_dir}",
        f"training_log={run_dir / 'logs' / f'{run_name}_run-3.log'}",
        f"checkpoint_model={run_dir / 'checkpoints' / f'{run_name}_run-3.model'}",
        f"compiled_model_if_available={run_dir / 'models' / f'{run_name}_compiled.model'}",
        "POLICY: Change exactly one scientific/training factor per iteration.",
        "POLICY: Set changed_parameter and change_note before every non-baseline run.",
        f"changed_parameter={args.changed_parameter}",
        f"change_note={args.change_note}",
        "",
        "Resolved wrapper hyperparameters:",
    ]
    for key, value in sorted(vars(args).items()):
        if not key.startswith("_"):
            lines.append(f"{key}={value}")
    lines.extend([
        "", "Resolved MACE command:", shlex.join(argv), "",
        "Analysis notes (append below; add a dated entry after every analysis or decision):",
        "- evidence: ",
        "- diagnosis / uncertainty: ",
        "- next one-factor test: ",
        "- why this test is informative: ",
        "- changed factor relative to this run: ",
        "",
    ])
    with destination.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def sha256(path: Path) -> str:
    """Return a content digest so a run is tied to its exact input files."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    """Best-effort Git revision; training must also work from exported sources."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(MLIP_DIR), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_metadata(args: argparse.Namespace, run_name: str, run_dir: Path, argv: list[str]) -> None:
    """Write immutable machine-readable provenance before optimization begins."""
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "run_name": run_name,
        "run_directory": str(run_dir),
        "git_revision": git_revision(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME"),
        "arguments": {key: str(value) for key, value in vars(args).items() if not key.startswith("_")},
        "mace_command": argv,
        "input_sha256": {
            "train_file": sha256(args.train_file),
            "valid_file": sha256(args.valid_file),
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def append_run_outcome(run_name: str, run_dir: Path, started: float, outcome: str, error: BaseException | None = None) -> None:
    """Record terminal status even when MACE raises before producing a model."""
    destination = SCRIPT_DIR / "hyperparameters.txt"
    lines = [
        f"RUN END {datetime.now().astimezone().isoformat()}",
        f"run_name={run_name}",
        f"run_directory={run_dir}",
        f"outcome={outcome}",
        f"elapsed_seconds={time.monotonic() - started:.1f}",
    ]
    if error is not None:
        lines.extend([f"error_type={type(error).__name__}", f"error_message={error}"])
    with destination.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(lines) + "\n")


def build_train_argv(args: argparse.Namespace, run_name: str, run_dir: Path) -> list[str]:
    train_file = require_file(args.train_file, "Missing target training extxyz")
    valid_file = require_file(args.valid_file, "Missing target validation extxyz")
    e0s = resolve_e0s(args.e0s)
    paths = {
        "model": run_dir / "models",
        "checkpoints": run_dir / "checkpoints",
        "log": run_dir / "logs",
        "results": run_dir / "results",
        "work": run_dir / "work",
    }
    argv = [
        "run_train.py",
        f"--name={run_name}",
        "--model=PolarMACE",
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
    argv.extend([f"--foundation_model={args.foundation_model}", "--multiheads_finetuning=False"])
    if args.restart_latest:
        argv.append("--restart_latest")
    if args.ema:
        argv.extend(["--ema", f"--ema_decay={args.ema_decay}"])
    if args.swa:
        argv.extend([
            "--swa",
            f"--start_swa={args.start_swa}",
            f"--swa_energy_weight={args.swa_energy_weight}",
            f"--swa_forces_weight={args.swa_forces_weight}",
        ])
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="polar1s_naive_orca_dft_e0")
    parser.add_argument("--foundation-model", default="polar-1-s")
    parser.add_argument("--train-file", type=Path, default=DATA_DIR / "target_train.xyz")
    parser.add_argument("--valid-file", type=Path, default=DATA_DIR / "target_valid.xyz")
    parser.add_argument("--runs-root", type=Path, default=RUNS_DIR,
                        help="Parent directory; each launch creates a unique timestamped child directory.")
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
    parser.add_argument("--max-num-epochs", type=int, default=100,
                        help="Active one-factor experiment length (default: 100).")
    parser.add_argument(
        "--status-every",
        type=int,
        default=250,
        help="Accepted for backward compatibility; current MACE controls progress logging internally.",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--default-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate from the validated 2026-09-04 baseline.")
    parser.add_argument(
        "--swa-lr",
        type=float,
        default=3e-4,
        help="Active one-factor Stage Two LR test (default: 3e-4).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--restart-latest", action="store_true")
    parser.add_argument("--ema", action="store_true", default=True, help="Enable EMA (baseline default).")
    parser.add_argument("--no-ema", action="store_false", dest="ema", help="Disable baseline EMA.")
    parser.add_argument("--ema-decay", type=float, default=0.99999)
    parser.add_argument("--start-swa", type=int, default=15)
    parser.add_argument("--swa-energy-weight", type=float, default=1.0)
    parser.add_argument("--swa-forces-weight", type=float, default=100_000.0)
    parser.add_argument("--swa", action="store_true", default=True, help="Enable baseline MACE Stage Two/SWA.")
    parser.add_argument("--no-swa", action="store_false", dest="swa", help="Disable baseline MACE Stage Two/SWA.")
    parser.add_argument("--changed-parameter", default="swa_learning_rate",
                        help="The one scientific/training factor changed from the preceding run.")
    parser.add_argument("--change-note", default=(
        "Set only Stage-Two LR to intermediate 3e-4 after 1e-4 underfit "
        "forces and 1e-3 caused energy drift."
    ),
                        help="Short rationale and expected effect for the single change.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


@record
def main() -> None:
    args = parse_args()
    if not MACE_REPO.is_dir():
        raise FileNotFoundError(f"Local MACE checkout is missing: {MACE_REPO}")
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))
    sys.path.insert(0, str(MACE_REPO.resolve()))
    run_name, run_dir = unique_run_context(args)
    sys.argv = build_train_argv(args, run_name, run_dir)
    if args.dry_run:
        print(shlex.join(sys.argv))
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("models", "checkpoints", "logs", "results", "work"):
        (run_dir / directory).mkdir()
    append_experiment_log(args, run_name, run_dir, sys.argv)
    write_run_metadata(args, run_name, run_dir, sys.argv)
    from mace.cli.run_train import main as run_train_main
    started = time.monotonic()
    try:
        run_train_main()
    except BaseException as error:
        append_run_outcome(run_name, run_dir, started, "failed", error)
        raise
    append_run_outcome(run_name, run_dir, started, "completed")


if __name__ == "__main__":
    main()
