#!/usr/bin/env python3
"""Run a job folder from mlip/codes on local, direct-GPU, or Slurm profiles.

Commands:
    python ~/mlip/exec.py --list-jobs
    python ~/mlip/exec.py -j 7_6_jaxoutputs -m pc
    python ~/mlip/exec.py -j 7_6_jaxoutputs -m viper-cpu --dry-run
    python ~/mlip/exec.py -j 6_26_NPT_MACE -m raven --entry NPTMACEbase.py --dry-run
    python ~/mlip/exec.py -j 6_26_NPT_MACE -m dungeon-gpu0 --entry expand/npt_r09_hot_w.py
    python ~/mlip/exec.py -j 6_26_NPT_MACE -m dungeon-gpu0 --entry expand/npt_r09_hot_w.py

    python ~/mlip/exec.py -j 6_26_NPT_MACE -m stormy-gpu0 --entry expand/npt_r09_hot_w7n1.py

Use the ~/mlip/exec.py path so the command works from any current directory.

Profiles:
    pc: run locally on this machine.
    dart-gpu0/dart-gpu1, dungeon-gpu0/dungeon-gpu1, stormy-gpu0/stormy-gpu1:
        run directly on that machine without Slurm, using CUDA_VISIBLE_DEVICES.
    raven, viper-cpu: write a Slurm wrapper in outputsfull/<run_id>/.

For non-PC machines, copy/sync the repo to that machine first. Direct-GPU profiles
execute immediately on that machine, even if --dry-run is present.

Use machines.local.json to add/override machine profiles; machines.example.json
contains a complete template.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODES = ROOT / "codes"
OUT = ROOT / "outputsfull"
ENTRIES = ("run.py", "main.py", "job.py")
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be an object keyed by machine name")
    return data


def machines() -> dict[str, dict]:
    result = load_json(ROOT / "machines.json")
    result.update(load_json(ROOT / "machines.local.json"))
    return {name: fill_defaults(name, profile) for name, profile in result.items()}


def fill_defaults(name: str, profile: dict) -> dict:
    if "root" not in profile:
        raise ValueError(f"Machine '{name}' is missing required field 'root'")
    scheduler = profile.get("scheduler", "local")
    if scheduler not in {"local", "slurm"}:
        raise ValueError(f"Machine '{name}' has invalid scheduler '{scheduler}'")
    return {
        "name": name,
        "root": str(profile["root"]),
        "codes_dir": str(profile.get("codes_dir", "codes")),
        "venv_activate": profile.get("venv_activate"),
        "scheduler": scheduler,
        "python": str(profile.get("python", "python")),
        "cpus": int(profile.get("cpus", 1)),
        "memory": str(profile.get("memory", "8G")),
        "walltime": str(profile.get("walltime", "01:00:00")),
        "module_lines": [str(line) for line in profile.get("module_lines", [])],
        "env_lines": [str(line) for line in profile.get("env_lines", [])],
    }


def jobs() -> list[str]:
    return sorted(path.name for path in CODES.iterdir() if path.is_dir()) if CODES.exists() else []


def entrypoint(job_dir: Path, entry: str | None) -> Path:
    if entry:
        path = job_dir / entry
        if path.is_file():
            return path
        raise FileNotFoundError(f"Entrypoint not found in job folder: {path}")

    candidates = [job_dir / name for name in ENTRIES] + [job_dir / f"{job_dir.name}.py"]
    candidates += sorted(job_dir.glob("*.py"))
    existing = [path for path in candidates if path.is_file()]
    unique = list(dict.fromkeys(existing))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(f"No Python entrypoint found in {job_dir}.")
    raise ValueError(
        f"Multiple Python files found in {job_dir}: "
        f"{', '.join(path.name for path in unique)}. Pass --entry <file.py>."
    )


def resolve_job(name: str, entry: str | None) -> tuple[Path, Path]:
    job_dir = CODES / name
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job folder not found: {job_dir}. Known jobs: {', '.join(jobs())}")
    return job_dir, entrypoint(job_dir, entry)


def paths(run_dir: Path, machine: dict) -> tuple[Path, str, str]:
    remote_name = "submit.slurm" if machine["scheduler"] == "slurm" else "run_remote.sh"
    command = "sbatch submit.slurm" if machine["scheduler"] == "slurm" else "bash run_remote.sh"
    return run_dir / ("run_local.ps1" if machine["name"] == "pc" else remote_name), remote_name, command


def write_outputs(run_dir: Path, job: str, job_dir: Path, entry: Path, machine: dict, dry_run: bool) -> None:
    rel_entry = entry.relative_to(job_dir).as_posix()
    stdout, stderr = run_dir / "raw_stdout.txt", run_dir / "raw_stderr.txt"
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "project_root": str(ROOT),
        "job": {"name": job, "directory": str(job_dir), "entry": str(entry), "entry_relative_to_job": rel_entry},
        "machine": machine,
        "outputs": {"directory": str(run_dir), "stdout": str(stdout), "stderr": str(stderr)},
    }, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_local.ps1").write_text(
        f'$ErrorActionPreference = "Stop"\n'
        f'$env:MLIP_ROOT = "{ROOT}"\n'
        f'$env:MLIP_OUTPUT_DIR = "{run_dir}"\n'
        f'Set-Location "{job_dir}"\n'
        f'& "{Path(sys.executable)}" "{rel_entry}" 1> "{stdout}" 2> "{stderr}"\n',
        encoding="utf-8",
    )
    if machine["scheduler"] == "slurm":
        write_remote(run_dir, job, rel_entry, machine)


def remote_paths(job: str, rel_entry: str, machine: dict) -> tuple[str, str]:
    remote_job = f'{machine["root"]}/{machine["codes_dir"]}/{job}'
    entry_path = Path(rel_entry)
    remote_cwd = f"{remote_job}/{entry_path.parent.as_posix()}" if entry_path.parent.as_posix() != "." else remote_job
    return remote_cwd, entry_path.name


def gpu_command(run_dir: Path, job: str, rel_entry: str, machine: dict) -> str:
    remote_cwd, remote_entry = remote_paths(job, rel_entry, machine)
    remote_out = f'{machine["root"]}/outputsfull/{run_dir.name}'
    setup = [f'source {machine["venv_activate"]}'] if machine["venv_activate"] else []
    return " && ".join([
        *setup,
        f'export MLIP_ROOT="{machine["root"]}"',
        f'export MLIP_OUTPUT_DIR="{remote_out}"',
        'mkdir -p "$MLIP_OUTPUT_DIR"',
        "export PYTHONNOUSERSITE=1",
        *machine["env_lines"],
        f'cd "{remote_cwd}"',
        f'{machine["python"]} "{remote_entry}"',
    ])


def write_remote(run_dir: Path, job: str, rel_entry: str, machine: dict) -> None:
    remote_out = f'{machine["root"]}/outputsfull/{run_dir.name}'
    remote_cwd, remote_entry = remote_paths(job, rel_entry, machine)
    setup = [f'source {machine["venv_activate"]}'] if machine["venv_activate"] else []
    lines = [
        "set -euo pipefail",
        "",
        f'export MLIP_ROOT="{machine["root"]}"',
        f'export MLIP_OUTPUT_DIR="{remote_out}"',
        'mkdir -p "$MLIP_OUTPUT_DIR"',
        *machine["module_lines"],
        *setup,
        "export PYTHONNOUSERSITE=1",
        *machine["env_lines"],
        f'cd "{remote_cwd}"',
        f'{machine["python"]} "{remote_entry}" > "$MLIP_OUTPUT_DIR/raw_stdout.txt" 2> "$MLIP_OUTPUT_DIR/raw_stderr.txt"',
    ]
    header = ["#!/bin/bash -l"]
    header += [
        f"#SBATCH --job-name={job[:32]}", "#SBATCH --ntasks=1",
        f'#SBATCH --cpus-per-task={machine["cpus"]}', f'#SBATCH --mem={machine["memory"]}',
        f'#SBATCH --time={machine["walltime"]}', "#SBATCH --output=slurm-%x-%j.out",
        "#SBATCH --error=slurm-%x-%j.err", "",
    ]
    (run_dir / "submit.slurm").write_text("\n".join([*header, *lines, ""]), encoding="utf-8")


def run_pc(run_dir: Path, job_dir: Path, entry: Path) -> int:
    env = os.environ | {"MLIP_ROOT": str(ROOT), "MLIP_OUTPUT_DIR": str(run_dir)}
    with (run_dir / "raw_stdout.txt").open("w", encoding="utf-8") as out:
        with (run_dir / "raw_stderr.txt").open("w", encoding="utf-8") as err:
            return subprocess.run(
                [sys.executable, entry.relative_to(job_dir).as_posix()],
                cwd=job_dir,
                env=env,
                stdout=out,
                stderr=err,
                check=False,
            ).returncode


def parser() -> argparse.ArgumentParser:
    names = sorted(machines())
    p = argparse.ArgumentParser(description="Run a folder from mlip/codes on a named machine profile.")
    p.add_argument("-j", "--job", help="Job folder under mlip/codes/.")
    p.add_argument("-m", "--machine", choices=names, default="pc", help="Machine profile to use.")
    p.add_argument("--entry", help="Python file inside the job folder. Required for ambiguous folders.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="For pc/slurm, create wrappers without running. Local GPU profiles run directly.",
    )
    p.add_argument("--list-jobs", action="store_true", help="List folders under mlip/codes and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.list_jobs:
        print("\n".join(jobs()) or "No jobs found under mlip/codes.")
        return 0
    if not args.job:
        raise SystemExit("--job is required unless --list-jobs is used")

    try:
        machine = machines()[args.machine]
        job_dir, entry = resolve_job(args.job, args.entry)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{args.job.replace(' ', '_')}_{args.machine}"
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    direct_gpu = args.machine != "pc" and machine["scheduler"] == "local"
    write_outputs(run_dir, args.job, job_dir, entry, machine, dry_run=args.dry_run and not direct_gpu)

    dry_wrapper, remote_name, remote_command = paths(run_dir, machine)
    print(f"Run id: {run_id}")
    print(f"Job: {args.job}")
    print(f"Entrypoint: {entry.relative_to(job_dir)}")
    print(f"Machine: {machine['name']}")
    print(f"Output directory: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")

    if direct_gpu:
        command = gpu_command(run_dir, args.job, entry.relative_to(job_dir).as_posix(), machine)
        print(f"Running GPU command: {command}")
        return subprocess.run(["bash", "-lc", command], check=False).returncode

    if args.dry_run:
        print(f"Dry run only. Wrapper: {dry_wrapper}")
        if args.machine != "pc":
            print(f"On {args.machine}, run: cd {run_dir.name} && {remote_command}")
        return 0

    if args.machine != "pc":
        print(f"Prepared remote wrapper: {run_dir / remote_name}")
        print("Copy/sync the repo to the target machine, then run this wrapper there.")
        return 0

    code = run_pc(run_dir, job_dir, entry)
    print(f"Raw stdout: {run_dir / 'raw_stdout.txt'}")
    print(f"Raw stderr: {run_dir / 'raw_stderr.txt'}")
    print(f"Exit code: {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
