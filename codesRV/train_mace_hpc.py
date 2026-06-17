#!/usr/bin/env python3
"""
HPC-friendly wrapper around mace.cli.run_train.main for fine-tuning PolarMACE.

Usage on Raven/Slurm is normally through submit_mace_polar_raven.slurm:
    sbatch submit_mace_polar_raven.slurm

You can override defaults by passing normal MACE CLI arguments after the script:
    python train_mace_hpc.py --max_num_epochs=2 --batch_size=1
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from mace.cli.run_train import main


def env(name: str, default: str) -> str:
    """Read a string setting from the environment."""
    return os.environ.get(name, default)


def env_path(name: str, default: str) -> str:
    """Read a path setting and return it as a POSIX string."""
    return str(Path(os.environ.get(name, default)).expanduser())


def build_default_args() -> list[str]:
    """Default arguments matching the user's working test-server command."""
    train_file = env_path(
        "MACE_TRAIN_FILE",
        "./ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz",
    )

    # For HPC compute nodes without internet, set this to a local model path, for example:
    # export MACE_FOUNDATION_MODEL=/u/$USER/mlip/models/MACE-POLAR-1-S.model
    foundation_model = env("MACE_FOUNDATION_MODEL", "polar-1-s")

    return [
        "--name", env("MACE_RUN_NAME", "polar_ft_1m"),
        "--model", "PolarMACE",
        "--foundation_model", foundation_model,
        "--train_file", train_file,
        "--valid_fraction", env("MACE_VALID_FRACTION", "0.05"),
        "--energy_key", env("MACE_ENERGY_KEY", "TotEnergy"),
        "--forces_key", env("MACE_FORCES_KEY", "force"),
        "--compute_forces", "True",
        "--E0s", env("MACE_E0S", "estimated"),
        "--loss", env("MACE_LOSS", "weighted"),
        "--stress_weight", env("MACE_STRESS_WEIGHT", "0.0"),
        "--force_mh_ft_lr", "True",
        "--default_dtype", env("MACE_DTYPE", "float32"),
        "--device", env("MACE_DEVICE", "cuda"),
        "--batch_size", env("MACE_BATCH_SIZE", "1"),
        "--max_num_epochs", env("MACE_MAX_EPOCHS", "20"),
        "--multiheads_finetuning", "True",
    ]


def main_wrapper() -> None:
    if len(sys.argv) == 1:
        args = build_default_args()
    else:
        # Allow normal MACE CLI usage: python train_mace_hpc.py --name ...
        args = sys.argv[1:]

    print("Running MACE with arguments:", flush=True)
    print(" ".join(shlex.quote(a) for a in args), flush=True)
    sys.argv = [sys.argv[0], *args]
    main()


if __name__ == "__main__":
    main_wrapper()
