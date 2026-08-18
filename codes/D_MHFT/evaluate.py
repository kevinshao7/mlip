#!/usr/bin/env python3
"""Evaluate a trained MACE model on a prepared extxyz file."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
MACE_REPO = MLIP_DIR / "mace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=SCRIPT_DIR / "data" / "target_test.xyz")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "data" / "predicted_test.xyz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head", default="Default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(MACE_REPO.resolve()))
    sys.argv = [
        "eval_configs.py",
        f"--configs={args.configs.resolve()}",
        f"--model={args.model.resolve()}",
        f"--output={args.output.resolve()}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        f"--device={args.device}",
        f"--head={args.head}",
    ]
    os.chdir(SCRIPT_DIR)
    from mace.cli.eval_configs import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
