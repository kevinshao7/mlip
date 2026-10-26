#!/usr/bin/env python3
"""
MACE-POLAR multihead replay fine-tuning using mace/cli/run_train.py directly.

This does not launch `mace_run_train` through subprocess. It configures the
arguments consumed by `mace.cli.run_train.main()` and then runs that module.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Use the PolarMACE-capable MACE checkout bundled with this workspace.
ROOT = Path(__file__).resolve().parent
MLIP_DIR = ROOT.parents[1]
MACE_REPO = MLIP_DIR / "mace"
CACHE_DIR = MLIP_DIR / "outputsfull" / ".cache"
if not MACE_REPO.is_dir():
    raise FileNotFoundError(f"Local MACE repository does not exist: {MACE_REPO}")
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
sys.path.insert(0, str(MACE_REPO))
sys.path.insert(0, str(MLIP_DIR / "codes"))
import finetune_megadataset as megadataset

import torch

from mace.cli.run_train import main as run_train_main


DATA_DIR = ROOT / "data"


def require_file(path: Path) -> str:
    """Return an absolute path string after checking that the file exists."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return str(path)


def main() -> None:
    train_file = require_file(DATA_DIR / "orca_train.xyz")
    valid_file = require_file(DATA_DIR / "orca_valid.xyz")
    replay_file = require_file(DATA_DIR / "polar_replay.xyz")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run MACE-POLAR training on CPU.")

    # Each attempt has an immutable output directory and a shared registry row.
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
    run_name = f"polar_orca_multihead_{stamp}"
    run_dir = ROOT / "runs" / run_name
    for name in ("models", "checkpoints", "logs", "results", "work"):
        (run_dir / name).mkdir(parents=True, exist_ok=False) if name == "models" else (run_dir / name).mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    # run_train.py parses its settings from sys.argv.
    sys.argv = [
        "run_train.py",
        f"--name={run_name}",
        "--model=PolarMACE",
        "--foundation_model=polar-1-m",
        f"--train_file={train_file}",
        f"--valid_file={valid_file}",
        f"--pt_train_file={replay_file}",
        "--multiheads_finetuning=True",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--total_charge_key=charge",
        "--total_spin_key=spin",
        "--E0s=estimated",
        "--energy_weight=1.0",
        "--forces_weight=10.0",
        "--stress_weight=0.0",
        "--weight_pt=1.0",
        "--batch_size=1",
        "--valid_batch_size=1",
        "--lr=0.0001",
        "--force_mh_ft_lr=True",
        "--ema",
        "--ema_decay=0.99999",
        "--default_dtype=float32",
        "--device=cuda",
        f"--model_dir={run_dir / 'models'}",
        f"--checkpoints_dir={run_dir / 'checkpoints'}",
        f"--log_dir={run_dir / 'logs'}",
        f"--results_dir={run_dir / 'results'}",
        f"--work_dir={run_dir / 'work'}",
    ]
    try:
        revision = subprocess.check_output(["git", "-C", str(MLIP_DIR), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = {"created_at": datetime.now().astimezone().isoformat(), "run_name": run_name, "run_directory": str(run_dir), "git_revision": revision,
                "arguments": {"foundation_model":"polar-1-m", "train_file":str(train_file), "valid_file":str(valid_file), "pt_train_file":str(replay_file), "method":"multihead_replay"}, "mace_command": sys.argv}
    metadata_path = run_dir / "run_metadata.json"; metadata_path.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    started = time.monotonic()
    try:
        run_train_main()
    except BaseException as error:
        megadataset.record_metadata_file(metadata_path, method="multihead_replay", wrapper="7_29_finetunepolar/runtrain.py", status="failed", ended_at=datetime.now().astimezone().isoformat(), duration_seconds=time.monotonic()-started, error=str(error))
        raise
    megadataset.record_metadata_file(metadata_path, method="multihead_replay", wrapper="7_29_finetunepolar/runtrain.py", status="completed", ended_at=datetime.now().astimezone().isoformat(), duration_seconds=time.monotonic()-started)


if __name__ == "__main__":
    main()
