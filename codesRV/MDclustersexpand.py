#!/usr/bin/env python3
"""Create CPU-only Slurm replicas of the MD cluster workflow.

This script duplicates the current MDclusters.py and MDclusters.slurm base files
into flat MDclusters0.py / MDclusters0.sh style files so that all jobs can be
submitted with: for f in *.sh; do sbatch "$f"; done.  Each generated MD script
writes to a separate output directory to avoid collisions when jobs run in
parallel.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "MDcluster_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate MDclusters.py and MDclusters.slurm for CPU sbatch runs."
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of replicas to create.")
    parser.add_argument(
        "--base-py",
        type=Path,
        default=PROJECT_ROOT / "MDclusters.py",
        help="Base MD Python script.",
    )
    parser.add_argument(
        "--base-slurm",
        type=Path,
        default=PROJECT_ROOT / "MDclusters.slurm",
        help="Base Slurm script.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory that will contain MDclusters0.py, MDclusters0.sh, ...",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=12345,
        help="First random seed used for Maxwell-Boltzmann velocities.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Optional CPU Slurm partition. If omitted, no partition line is written.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="CPUs requested per replica.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit each generated Slurm script with sbatch.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required base file not found: {path}")
    return path


def replace_one(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise ValueError(f"Could not find expected text while patching {label}: {old!r}")
    return text.replace(old, new, 1)


def patch_md_script(base_text: str, run_id: str, seed: int) -> str:
    """Patch one copied MDclusters.py for a unique CPU-only replica."""
    text = base_text

    # Generated files live in MDcluster_runs by default, so the project root is
    # one directory above the generated script.
    text = re.sub(
        r"^PROJECT_ROOT\s*=.*$",
        "PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^MD_RESULTS_DIR\s*=.*$",
        f'MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "MDresults", "{run_id}")',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # Keep generated runs CPU-only even if the base file is later switched back
    # to CUDA for an interactive test.
    text = re.sub(r'device\s*=\s*["\']cuda["\']', 'device="cpu"', text)

    text = replace_one(
        text,
        "def simpleMD(init_conf, temp, calc, fname, s, T):",
        "def simpleMD(init_conf, temp, calc, fname, s, T, random_seed=None):",
        "simpleMD signature",
    )
    text = replace_one(
        text,
        "MaxwellBoltzmannDistribution(init_conf, temperature_K=temp)",
        (
            "MaxwellBoltzmannDistribution(\n"
            "        init_conf,\n"
            "        temperature_K=temp,\n"
            "        rng=np.random.default_rng(random_seed),\n"
            "    )"
        ),
        "velocity initialization",
    )

    text = re.sub(
        r"(simpleMD\([\s\S]*?\n\s*T\s*=\s*[^,\n]+,\n)(\s*\))",
        rf"\1    random_seed={seed},\n\2",
        text,
        count=1,
    )
    return text


def non_cuda_module_lines(base_slurm_text: str) -> list[str]:
    module_lines: list[str] = []
    for line in base_slurm_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("module load "):
            continue
        if "cuda" in stripped.lower():
            continue
        module_lines.append(line)
    return module_lines


def find_line(base_slurm_text: str, prefix: str) -> str | None:
    for line in base_slurm_text.splitlines():
        if line.strip().startswith(prefix):
            return line
    return None


def patch_slurm_script(
    base_slurm_text: str,
    run_id: str,
    py_name: str,
    partition: str | None,
    cpus_per_task: int,
) -> str:
    """Build a CPU-only Slurm script while preserving useful base settings."""
    mem_line = find_line(base_slurm_text, "#SBATCH --mem=") or "#SBATCH --mem=64G"
    time_line = find_line(base_slurm_text, "#SBATCH --time=") or "#SBATCH --time=05:00:00"
    ntasks_line = find_line(base_slurm_text, "#SBATCH --ntasks=") or "#SBATCH --ntasks=1"
    source_line = find_line(base_slurm_text, "source ")
    path_line = find_line(base_slurm_text, "export PATH=")
    module_lines = non_cuda_module_lines(base_slurm_text)

    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name=mace_MD_{run_id}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines.extend(
        [
            ntasks_line,
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            mem_line,
            time_line,
            "#SBATCH --output=slurm-%x-%j.out",
            "#SBATCH --error=slurm-%x-%j.err",
            "",
            "set -e",
            "",
            "module purge",
        ]
    )
    lines.extend(module_lines)
    if source_line:
        lines.extend(["", source_line])
    lines.extend(
        [
            "export PYTHONNOUSERSITE=1",
        ]
    )
    if path_line:
        lines.append(path_line)
    lines.extend(
        [
            "",
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}",
            "export CUDA_VISIBLE_DEVICES=",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'cd "${SCRIPT_DIR}/.."',
            "",
            'echo "Job started on $(hostname)"',
            'echo "Project directory: $(pwd)"',
            'echo "Generated script directory: ${SCRIPT_DIR}"',
            "",
            f'srun python "${{SCRIPT_DIR}}/{py_name}"',
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.cpus_per_task <= 0:
        raise ValueError("--cpus-per-task must be positive")

    base_py = require_file(args.base_py)
    base_slurm = require_file(args.base_slurm)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_py_text = base_py.read_text(encoding="utf-8")
    base_slurm_text = base_slurm.read_text(encoding="utf-8")

    generated_slurms: list[Path] = []
    for index in range(args.runs):
        run_id = str(index)
        result_id = f"run_{index:03d}"
        seed = args.seed_start + index

        py_name = f"MDclusters{run_id}.py"
        slurm_name = f"MDclusters{run_id}.sh"
        py_path = out_dir / py_name
        slurm_path = out_dir / slurm_name

        write_text(py_path, patch_md_script(base_py_text, result_id, seed), args.overwrite)
        write_text(
            slurm_path,
            patch_slurm_script(
                base_slurm_text,
                run_id=result_id,
                py_name=py_name,
                partition=args.partition,
                cpus_per_task=args.cpus_per_task,
            ),
            args.overwrite,
        )
        generated_slurms.append(slurm_path)
        print(f"wrote {display_path(py_path)}")
        print(f"wrote {display_path(slurm_path)}")

    if args.submit:
        for slurm_path in generated_slurms:
            print(f"submitting {display_path(slurm_path)}")
            subprocess.run(["sbatch", slurm_path.name], cwd=slurm_path.parent, check=True)
    else:
        print("")
        print("Generated CPU-only Slurm replicas. Submit with:")
        print(f"  cd {display_path(out_dir)}")
        print('  for f in *.sh; do sbatch "$f"; done')


if __name__ == "__main__":
    main()
