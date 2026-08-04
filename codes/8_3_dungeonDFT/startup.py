#!/usr/bin/env python3
"""Run the 8_1/8_2 ORCA input files in serial halves.

Examples:
    python startup.py --id 1
    python startup.py --id 2 --resume
    python startup.py --id 1 --orca-command /path/to/orca_qc

Outputs and ORCA side files are written under:
    outputsfull/8_3_dungeonDFT/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
OUTPUT_DIR = MLIP_DIR / "outputsfull" / "8_3_dungeonDFT"

SOURCE_DIRS = (
    MLIP_DIR / "codes" / "8_1_viperDFT",
    MLIP_DIR / "codes" / "8_2_ravenDFT",
)
INPUT_PATTERNS = (
    "r09_hot_w_h2training_first100_*.inp",
    "r09_hot_w_h2training_last100_*.inp",
)
EXPECTED_PER_SOURCE = 100
EXPECTED_TOTAL = 200
BASIS_FILE = "def2-tzvpd.bas"
THREADS = "24"
DEFAULT_ORCA_COMMAND = "orca_qc"

FINAL_ENERGY_MARKER = "FINAL SINGLE POINT ENERGY"
NORMAL_TERMINATION_MARKER = "ORCA TERMINATED NORMALLY"


@dataclass(frozen=True)
class OrcaInput:
    index: int
    source_dir: Path
    inp_path: Path
    basis_path: Path

    @property
    def stem(self) -> str:
        return self.inp_path.stem

    @property
    def out_path(self) -> Path:
        return OUTPUT_DIR / f"{self.stem}.out"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def discover_inputs() -> list[OrcaInput]:
    jobs: list[OrcaInput] = []
    basis_hashes: dict[Path, str] = {}

    if len(SOURCE_DIRS) != len(INPUT_PATTERNS):
        fail("Internal configuration error: SOURCE_DIRS and INPUT_PATTERNS lengths differ")

    for source_dir, pattern in zip(SOURCE_DIRS, INPUT_PATTERNS):
        if not source_dir.is_dir():
            fail(f"Source directory is missing: {source_dir}")

        basis_path = source_dir / BASIS_FILE
        if not basis_path.is_file():
            fail(f"Required ORCA basis file is missing: {basis_path}")
        basis_hashes[basis_path] = sha256(basis_path)

        inputs = sorted(source_dir.glob(pattern))
        if len(inputs) != EXPECTED_PER_SOURCE:
            fail(f"Expected {EXPECTED_PER_SOURCE} inputs matching {source_dir / pattern}, found {len(inputs)}")

        for inp_path in inputs:
            validate_input_text(inp_path)
            jobs.append(OrcaInput(len(jobs) + 1, source_dir, inp_path, basis_path))

    if len(jobs) != EXPECTED_TOTAL:
        fail(f"Expected {EXPECTED_TOTAL} total ORCA inputs, found {len(jobs)}")

    stems = [job.stem for job in jobs]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        fail(f"Duplicate input stems would collide in {OUTPUT_DIR}: {', '.join(duplicate_stems)}")

    if len(set(basis_hashes.values())) != 1:
        details = ", ".join(f"{path}: {digest}" for path, digest in basis_hashes.items())
        fail(f"Basis files differ across source directories: {details}")

    return jobs


def validate_input_text(inp_path: Path) -> None:
    text = inp_path.read_text(encoding="utf-8")
    if not text.strip():
        fail(f"Input file is empty: {inp_path}")
    if "*xyz" not in text:
        fail(f"Input file does not contain an *xyz charge/multiplicity block: {inp_path}")
    if f'GTOName "{BASIS_FILE}"' not in text:
        fail(f'Input file does not reference GTOName "{BASIS_FILE}": {inp_path}')


def selected_jobs(jobs: list[OrcaInput], run_id: int) -> list[OrcaInput]:
    base, remainder = divmod(len(jobs), 2)
    sizes = [base + (1 if i < remainder else 0) for i in range(2)]
    start = sum(sizes[: run_id - 1])
    stop = start + sizes[run_id - 1]
    return jobs[start:stop]


def parse_output_status(out_path: Path) -> tuple[bool, bool]:
    if not out_path.is_file():
        return False, False
    text = out_path.read_text(encoding="utf-8", errors="replace")
    has_energy = FINAL_ENERGY_MARKER in text
    terminated = NORMAL_TERMINATION_MARKER in text
    return has_energy, terminated


def prepare_outputs(jobs: list[OrcaInput], resume: bool, force: bool) -> None:
    if resume and force:
        fail("--resume and --force are mutually exclusive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        if not job.out_path.exists():
            continue

        has_energy, terminated = parse_output_status(job.out_path)
        if resume and has_energy and terminated:
            continue
        if force:
            job.out_path.unlink()
            continue
        if resume:
            fail(
                f"Existing output is incomplete or failed: {job.out_path}. "
                "Inspect it, remove it, or rerun with --force."
            )
        fail(f"Output already exists: {job.out_path}. Use --resume to skip completed outputs or --force to overwrite.")


def orca_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = THREADS
    env["MKL_NUM_THREADS"] = THREADS
    env["OPENBLAS_NUM_THREADS"] = THREADS
    return env


def validate_orca_command(orca_command: str) -> str:
    resolved = shutil.which(orca_command)
    if resolved is None:
        fail(
            f"ORCA executable was not found on PATH: {orca_command!r}. "
            "Load the ORCA module first or pass --orca-command /path/to/orca_qc."
        )

    resolved_path = Path(resolved).resolve()
    if resolved_path in (Path("/usr/bin/orca"), Path("/bin/orca")):
        fail(
            f"{orca_command!r} resolves to {resolved_path}, which is the GNOME screen reader, "
            "not the ORCA quantum chemistry executable. Load the ORCA module first or pass "
            "--orca-command /path/to/orca_qc."
        )
    return str(resolved_path)


def stage_basis_file(job: OrcaInput) -> None:
    shutil.copy2(job.basis_path, OUTPUT_DIR / BASIS_FILE)


def should_skip_completed(job: OrcaInput, resume: bool) -> bool:
    if not resume:
        return False
    has_energy, terminated = parse_output_status(job.out_path)
    if has_energy and terminated:
        print(f"Skipping completed {job.stem}", flush=True)
        return True
    return False


def run_job(job: OrcaInput, env: dict[str, str], orca_command: str) -> None:
    stage_basis_file(job)
    cmd = [orca_command, str(job.inp_path)]
    print(f"[{job.index:03d}/{EXPECTED_TOTAL}] Running {job.inp_path.name} -> {job.out_path.name}", flush=True)

    with job.out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        process = subprocess.Popen(
            cmd,
            cwd=OUTPUT_DIR,
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
        fail(f"ORCA failed for {job.inp_path} with exit code {return_code}; output: {job.out_path}")

    has_energy, terminated = parse_output_status(job.out_path)
    if not has_energy:
        fail(f"ORCA output is missing {FINAL_ENERGY_MARKER!r}: {job.out_path}")
    if not terminated:
        fail(f"ORCA output is missing {NORMAL_TERMINATION_MARKER!r}: {job.out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id",
        type=int,
        choices=(1, 2),
        required=True,
        help="Serial shard to run: 1=first half, 2=second half.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip outputs that already contain a final single point energy and normal ORCA termination.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs for the selected shard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected inputs without running ORCA.",
    )
    parser.add_argument(
        "--orca-command",
        default=DEFAULT_ORCA_COMMAND,
        help=f"ORCA executable command. Default: {DEFAULT_ORCA_COMMAND}",
    )
    args = parser.parse_args()

    jobs = discover_inputs()
    shard = selected_jobs(jobs, args.id)
    if not shard:
        fail(f"No inputs selected for --id {args.id}")

    first = shard[0].inp_path.name
    last = shard[-1].inp_path.name
    print(f"Selected {len(shard)} of {len(jobs)} ORCA inputs for --id {args.id}: {first} through {last}")
    print(f"Output directory: {OUTPUT_DIR}")

    if args.dry_run:
        for job in shard:
            print(f"{job.index:03d} {job.inp_path}")
        return

    orca_command = validate_orca_command(args.orca_command)
    prepare_outputs(shard, resume=args.resume, force=args.force)

    env = orca_env()
    for job in shard:
        if should_skip_completed(job, resume=args.resume):
            continue
        run_job(job, env, orca_command)

    print(f"Completed shard --id {args.id}: {len(shard)} inputs checked/run successfully")


if __name__ == "__main__":
    main()
