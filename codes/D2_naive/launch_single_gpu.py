#!/usr/bin/env python3
"""Launch naive PolarMACE fine tuning on one selected GPU.

Use ``--dry-run`` to inspect the command without starting training. Any
arguments following ``--`` are forwarded to ``trainmace.py``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="Physical GPU ID to use (default: 0).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to trainmace.py; prefix them with '--'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gpu.strip() or "," in args.gpu:
        raise ValueError("--gpu must name exactly one physical GPU ID, e.g. --gpu 0")

    forwarded = args.train_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    command = [sys.executable, str(SCRIPT_DIR / "trainmace.py"), *forwarded]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
    print(f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}")
    print(shlex.join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=SCRIPT_DIR, env=environment, check=True)


if __name__ == "__main__":
    main()
