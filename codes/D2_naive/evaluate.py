#!/usr/bin/env python3
"""Evaluate a naive-fine-tuned MACE model on prepared ORCA extxyz data."""

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
    # The current MACE evaluator preserves REF_energy and REF_forces already
    # stored in extxyz; its older --energy_key/--forces_key flags were removed.
    sys.argv = ["eval_configs.py", f"--configs={args.configs.resolve()}", f"--model={args.model.resolve()}", f"--output={args.output.resolve()}", "--default_dtype=float32", f"--device={args.device}", f"--head={args.head}"]
    os.chdir(SCRIPT_DIR)
    from mace.cli.eval_configs import main as eval_main
    eval_main()


if __name__ == "__main__":
    main()
