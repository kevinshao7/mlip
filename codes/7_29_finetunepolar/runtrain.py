#!/usr/bin/env python3
"""
MACE-POLAR multihead replay fine-tuning using mace/cli/run_train.py directly.

This does not launch `mace_run_train` through subprocess. It configures the
arguments consumed by `mace.cli.run_train.main()` and then runs that module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Use the PolarMACE-capable MACE checkout bundled with this workspace.
ROOT = Path(__file__).resolve().parent
MACE_REPO = ROOT.parents[1] / "mace"
CACHE_DIR = ROOT.parents[1] / "outputsfull" / ".cache"
if not MACE_REPO.is_dir():
    raise FileNotFoundError(f"Local MACE repository does not exist: {MACE_REPO}")
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
sys.path.insert(0, str(MACE_REPO))

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

    # Keep logs, checkpoints, and model outputs relative to this script.
    os.chdir(ROOT)

    # run_train.py parses its settings from sys.argv.
    sys.argv = [
        "run_train.py",
        "--name=polar_orca_multihead",
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
    ]

    run_train_main()


if __name__ == "__main__":
    main()
