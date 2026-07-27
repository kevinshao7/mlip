#!/usr/bin/env python3
"""Run either half of the ORCA H2-formation focused-frame inputs in series.

The launcher loads the MPI module before each ORCA call:
    module load mpi/openmpi-x86_64

Examples:
    python startup.py --batch first
    python startup.py --batch last

Equivalent numeric switches:
    python startup.py --run 0   # frames 001-010
    python startup.py --run 1   # frames 011-020
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATTERN = "r09_hot_w_h2formation_frame_*.inp"
EXPECTED_INPUTS = 20
THREADS = "24"
BASIS_FILE = "def2-tzvpd.bas"
RUNTIME_DIR = SCRIPT_DIR / ".orca_runtime"
DEFAULT_MPI_MODULE = "mpi/openmpi-x86_64"
ORCA_COMMAND = "orca_qc"


def selected_inputs(batch: str) -> list[Path]:
    inputs = sorted(SCRIPT_DIR.glob(INPUT_PATTERN))
    if len(inputs) != EXPECTED_INPUTS:
        raise SystemExit(f"Expected {EXPECTED_INPUTS} ORCA inputs matching {INPUT_PATTERN}, found {len(inputs)}")
    return inputs[:10] if batch == "first" else inputs[10:]


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
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return out_path


def orca_command(run_path: Path, mpi_module: str | None) -> list[str]:
    if os.name == "posix" and mpi_module:
        shell_command = (
            f"module load {shlex.quote(mpi_module)} && "
            f"exec {shlex.quote(ORCA_COMMAND)} {shlex.quote(str(run_path))}"
        )
        return ["bash", "-lc", shell_command]
    return [ORCA_COMMAND, str(run_path)]


def run_input(inp_path: Path, env: dict[str, str], mpi_module: str | None) -> None:
    basis = basis_path()
    run_path = runtime_input(inp_path, basis)
    out_path = inp_path.with_suffix(".out")
    print(f"Running {inp_path.name} -> {out_path.name}", flush=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        process = subprocess.Popen(
            orca_command(run_path, mpi_module),
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


def resolve_batch(args: argparse.Namespace) -> str:
    if args.batch is not None:
        return args.batch
    if args.run is not None:
        return "first" if args.run == 0 else "last"
    raise SystemExit("Choose one batch with --batch first, --batch last, --run 0, or --run 1.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("first", "last"), help="first runs frames 001-010; last runs 011-020.")
    parser.add_argument("--run", type=int, choices=(0, 1), help="0 runs frames 001-010; 1 runs frames 011-020.")
    parser.add_argument(
        "--mpi-module",
        default=DEFAULT_MPI_MODULE,
        help=f"MPI module to load before each ORCA run; use an empty string to skip. Default: {DEFAULT_MPI_MODULE}",
    )
    args = parser.parse_args()

    batch = resolve_batch(args)
    env = orca_env()
    mpi_module = args.mpi_module or None
    for inp_path in selected_inputs(batch):
        run_input(inp_path, env, mpi_module)


if __name__ == "__main__":
    main()
