#!/usr/bin/env python3
"""Run either half of the ORCA large-cluster inputs in series.
module load mpi/openmpi-x86_64
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
BASIS_FILE = "def2-tzvpd.bas"
RUNTIME_DIR = SCRIPT_DIR / ".orca_runtime"


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


def basis_path() -> Path:
    path = SCRIPT_DIR / BASIS_FILE
    if not path.is_file():
        raise SystemExit(f"Required ORCA basis file is missing: {path}")
    return path.resolve()


def runtime_input(inp_path: Path, basis: Path) -> Path:
    RUNTIME_DIR.mkdir(exist_ok=True)
    text = inp_path.read_text(encoding="utf-8")
    text = text.replace(f'GTOName "{BASIS_FILE}"', f'GTOName "{basis}"')
    out_path = RUNTIME_DIR / inp_path.name
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return out_path


def run_input(inp_path: Path, env: dict[str, str]) -> None:
    basis = basis_path()
    run_path = runtime_input(inp_path, basis)
    out_path = inp_path.with_suffix(".out")
    print(f"Running {inp_path.name} -> {out_path.name}", flush=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        process = subprocess.Popen(
            ["orca_qc", str(run_path)],
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
