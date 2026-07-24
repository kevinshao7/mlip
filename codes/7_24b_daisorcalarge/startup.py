#!/usr/bin/env python3
"""Run either half of the ORCA large-cluster inputs in series.

Examples:
    python startup.py --run 0
    python startup.py --run 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATTERN = "r09_hot_w_large_cluster_*.inp"
THREADS = "24"


def selected_inputs(run: int) -> list[Path]:
    inputs = sorted(SCRIPT_DIR.glob(INPUT_PATTERN))
    if len(inputs) != 30:
        raise SystemExit(f"Expected 30 ORCA inputs matching {INPUT_PATTERN}, found {len(inputs)}")
    return inputs[:15] if run == 0 else inputs[15:]


def orca_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = THREADS
    env["MKL_NUM_THREADS"] = THREADS
    env["OPENBLAS_NUM_THREADS"] = THREADS
    return env


def run_input(inp_path: Path, env: dict[str, str]) -> None:
    out_path = inp_path.with_suffix(".out")
    print(f"Running {inp_path.name} -> {out_path.name}", flush=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        process = subprocess.Popen(
            ["orca", str(inp_path)],
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            out_handle.write(line)

    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"ORCA failed for {inp_path.name} with exit code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=int,
        choices=(0, 1),
        required=True,
        help="0 runs clusters 001-015; 1 runs clusters 016-030.",
    )
    args = parser.parse_args()

    env = orca_env()
    for inp_path in selected_inputs(args.run):
        run_input(inp_path, env)


if __name__ == "__main__":
    main()
