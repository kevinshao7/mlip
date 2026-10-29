#!/usr/bin/env python3
"""Run the four D4rev atomic ORCA jobs sequentially with orca_qc."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from generate_atom_inputs import ATOM_SPECS, ORCA_NPROCS, RUNS_DIR, check_inputs


ENERGY_MARKER = "FINAL SINGLE POINT ENERGY"
NORMAL_MARKER = "ORCA TERMINATED NORMALLY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orca-command",
        default=os.environ.get("ORCA_COMMAND", "orca_qc"),
        help="ORCA executable or wrapper name (default: ORCA_COMMAND or orca_qc).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip normally completed outputs; still reject incomplete outputs.",
    )
    return parser.parse_args()


def output_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return ENERGY_MARKER in text and NORMAL_MARKER in text


def orca_environment() -> dict[str, str]:
    env = os.environ.copy()
    mpi_bin = Path(env.get("MPI_BIN_DIR", "/usr/lib64/openmpi/bin"))
    mpi_lib = Path(env.get("MPI_LIB_DIR", "/usr/lib64/openmpi/lib"))
    if mpi_bin.is_dir():
        env["PATH"] = f"{mpi_bin}{os.pathsep}{env.get('PATH', '')}"
    if mpi_lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{mpi_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"

    # ORCA supplies parallelism through 20 MPI ranks. Keep linked numerical
    # libraries single-threaded so each rank does not create another pool.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def run_one(executable: str, atom: str, env: dict[str, str]) -> None:
    run_dir = RUNS_DIR / atom
    stem = f"orcaatomizationD4rev{atom}"
    input_path = run_dir / f"{stem}.inp"
    output_path = run_dir / f"{stem}.out"

    print(
        f"Starting {atom}: {executable} {input_path.name} "
        f"({ORCA_NPROCS} MPI ranks)",
        flush=True,
    )
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        process = subprocess.Popen(
            [executable, input_path.name],
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"ORCA exited with code {return_code} for {atom}; retained {output_path} for diagnosis"
        )
    if not output_is_complete(output_path):
        raise RuntimeError(
            f"ORCA output for {atom} lacks the final energy or normal-termination marker: {output_path}"
        )
    print(f"Completed {atom}: {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    check_inputs()
    executable = shutil.which(args.orca_command)
    if executable is None:
        raise FileNotFoundError(f"ORCA command is not available: {args.orca_command}")
    env = orca_environment()
    if shutil.which("mpirun", path=env.get("PATH")) is None:
        raise FileNotFoundError("mpirun is unavailable after applying MPI_BIN_DIR")

    for spec in ATOM_SPECS:
        output_path = RUNS_DIR / spec.symbol / f"orcaatomizationD4rev{spec.symbol}.out"
        if output_path.exists():
            if args.resume and output_is_complete(output_path):
                print(f"Skipping completed {spec.symbol}: {output_path}", flush=True)
                continue
            state = "incomplete" if not output_is_complete(output_path) else "existing"
            raise FileExistsError(
                f"Refusing to overwrite {state} ORCA output: {output_path}. "
                "Archive it first, or use --resume to skip normally completed outputs."
            )
        run_one(executable, spec.symbol, env)


if __name__ == "__main__":
    main()
