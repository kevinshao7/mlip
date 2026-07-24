#!/usr/bin/env python3
"""Submit either half of the DAIS ORCA large-cluster jobs in series.

Examples:
    python startup.py --run 0
    python startup.py --run 1
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
JOB_PATTERN = "r09_hot_w_large_cluster_*.slurm"
SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def selected_jobs(run: int) -> list[Path]:
    jobs = sorted(SCRIPT_DIR.glob(JOB_PATTERN))
    if len(jobs) != 30:
        raise SystemExit(f"Expected 30 Slurm files matching {JOB_PATTERN}, found {len(jobs)}")
    return jobs[:15] if run == 0 else jobs[15:]


def submit_job(job: Path, dependency_job_id: str | None) -> str:
    command = ["sbatch"]
    if dependency_job_id is not None:
        command.append(f"--dependency=afterany:{dependency_job_id}")
    command.append(str(job))

    result = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    match = SBATCH_JOB_ID_RE.search(output)
    if match is None:
        raise SystemExit(f"Could not parse sbatch job id from output: {output!r}")
    job_id = match.group(1)
    print(f"{job.name}: {output}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=int,
        choices=(0, 1),
        required=True,
        help="0 submits clusters 001-015; 1 submits clusters 016-030.",
    )
    args = parser.parse_args()

    dependency_job_id = None
    for job in selected_jobs(args.run):
        dependency_job_id = submit_job(job, dependency_job_id)


if __name__ == "__main__":
    main()
