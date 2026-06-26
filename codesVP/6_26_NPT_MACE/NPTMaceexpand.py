#!/usr/bin/env python3
"""Generate Uranus-profile MACE MD scripts and Slurm launchers.

This implements the condition grid sketched in the original pseudocode:

    3 Uranus temperature profiles
    x 3 compositions
    x source rows 2..10 from the Uranus profile table

Each generated Python script is a patched copy of ``NPTMACEbase.py`` with a
specific density, target temperature, composition, output directory, trajectory
name, save interval, and number of MD steps.  The generated Slurm scripts live
next to the generated Python files in ``expand/``.

The generated scripts preserve the base workflow: brief Langevin NVT
thermalization followed by NPT Berendsen integration.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "expand"
DEFAULT_PROFILE_CSV = REPO_ROOT / "codesDG" / "uranus_profiles.csv"

DEFAULT_SOURCE_ROW_START = 2
DEFAULT_SOURCE_ROW_END = 10
DEFAULT_SAVE_INTERVAL_STEPS = 100
DEFAULT_MD_STEPS = 10_000

TEMPERATURE_PROFILES = {
    "hot": "hot_uranus_temperature_K",
    "cold": "cold_uranus_temperature_K",
    "pref": "preferred_uranus_temperature_K",
}

COMPOSITIONS = {
    "w": {
        "label": "pure H2O",
        "solvent_line": (
            "simbox.add_solvent([water], ratio=[1], zdim=boxsize, "
            "density=densitygcm3)"
        ),
    },
    "w4n1": {
        "label": "H2O:NH3 = 4:1",
        "solvent_line": (
            "simbox.add_solvent([water, amm], ratio=[4, 1], zdim=boxsize, "
            "density=densitygcm3)"
        ),
    },
    "w7n1": {
        "label": "H2O:NH3 = 7:1",
        "solvent_line": (
            "simbox.add_solvent([water, amm], ratio=[7, 1], zdim=boxsize, "
            "density=densitygcm3)"
        ),
    },
}

FALLBACK_PROFILE_ROWS = [
    {
        "source_row": 2,
        "pressure_GPa": 0.0001,
        "density_g_cm3": 0.000449,
        "hot_uranus_temperature_K": 76,
        "cold_uranus_temperature_K": 76,
        "preferred_uranus_temperature_K": 76,
    },
    {
        "source_row": 3,
        "pressure_GPa": 0.0011,
        "density_g_cm3": 0.00281,
        "hot_uranus_temperature_K": 136,
        "cold_uranus_temperature_K": 156,
        "preferred_uranus_temperature_K": 179,
    },
    {
        "source_row": 4,
        "pressure_GPa": 0.01,
        "density_g_cm3": 0.012,
        "hot_uranus_temperature_K": 269,
        "cold_uranus_temperature_K": 269,
        "preferred_uranus_temperature_K": 398,
    },
    {
        "source_row": 5,
        "pressure_GPa": 0.11,
        "density_g_cm3": 0.0495,
        "hot_uranus_temperature_K": 537,
        "cold_uranus_temperature_K": 481,
        "preferred_uranus_temperature_K": 759,
    },
    {
        "source_row": 6,
        "pressure_GPa": 1,
        "density_g_cm3": 0.14,
        "hot_uranus_temperature_K": 1020,
        "cold_uranus_temperature_K": 854,
        "preferred_uranus_temperature_K": 1240,
    },
    {
        "source_row": 7,
        "pressure_GPa": 10,
        "density_g_cm3": 0.344,
        "hot_uranus_temperature_K": 2050,
        "cold_uranus_temperature_K": 1500,
        "preferred_uranus_temperature_K": 1920,
    },
    {
        "source_row": 8,
        "pressure_GPa": 15,
        "density_g_cm3": 0.405,
        "hot_uranus_temperature_K": 2340,
        "cold_uranus_temperature_K": 1640,
        "preferred_uranus_temperature_K": 2070,
    },
    {
        "source_row": 9,
        "pressure_GPa": 15,
        "density_g_cm3": 1.19,
        "hot_uranus_temperature_K": 2340,
        "cold_uranus_temperature_K": 1640,
        "preferred_uranus_temperature_K": 2070,
    },
    {
        "source_row": 10,
        "pressure_GPa": 100,
        "density_g_cm3": 3.72,
        "hot_uranus_temperature_K": 5520,
        "cold_uranus_temperature_K": 1920,
        "preferred_uranus_temperature_K": 2840,
    },
    {
        "source_row": 11,
        "pressure_GPa": 550,
        "density_g_cm3": 4.07,
        "hot_uranus_temperature_K": 6080,
        "cold_uranus_temperature_K": 2200,
        "preferred_uranus_temperature_K": 5460,
    },
    {
        "source_row": 12,
        "pressure_GPa": 550,
        "density_g_cm3": 9.08,
        "hot_uranus_temperature_K": 6080,
        "cold_uranus_temperature_K": 2200,
        "preferred_uranus_temperature_K": 5460,
    },
    {
        "source_row": 13,
        "pressure_GPa": 820,
        "density_g_cm3": 10.3,
        "hot_uranus_temperature_K": 6080,
        "cold_uranus_temperature_K": 2210,
        "preferred_uranus_temperature_K": 7160,
    },
]


@dataclass(frozen=True)
class Condition:
    source_row: int
    pressure_gpa: float
    density_g_cm3: float
    profile_slug: str
    profile_column: str
    composition_slug: str
    composition_label: str
    solvent_line: str
    temperature_k: float

    @property
    def run_id(self) -> str:
        return f"r{self.source_row:02d}_{self.profile_slug}_{self.composition_slug}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Uranus-profile MACE MD scripts and Slurm files."
    )
    parser.add_argument(
        "--base-py",
        type=Path,
        default=SCRIPT_DIR / "NPTMACEbase.py",
        help="Base Python MD script to patch.",
    )
    parser.add_argument(
        "--base-slurm",
        type=Path,
        default=SCRIPT_DIR / "NPTMACEbase.slurm",
        help="Base Slurm script to patch.",
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=DEFAULT_PROFILE_CSV,
        help="Wide Uranus profile CSV. If absent, built-in Table 4 values are used.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated Python and Slurm files.",
    )
    parser.add_argument(
        "--source-row-start",
        type=int,
        default=DEFAULT_SOURCE_ROW_START,
        help="First Uranus source_row to include.",
    )
    parser.add_argument(
        "--source-row-end",
        type=int,
        default=DEFAULT_SOURCE_ROW_END,
        help="Last Uranus source_row to include, inclusive.",
    )
    parser.add_argument(
        "--save-interval-steps",
        type=int,
        default=DEFAULT_SAVE_INTERVAL_STEPS,
        help="Trajectory/thermo write interval patched into simpleMD(s=...).",
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=DEFAULT_MD_STEPS,
        help="Total MD steps patched into simpleMD(T=...).",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=24,
        help="CPUs requested per generated Slurm job and used by torch/OpenMP.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Optional Slurm partition. If omitted, no partition line is written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit generated Slurm scripts with sbatch.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required base file not found: {path}")
    return path


def read_profile_rows(profile_csv: Path) -> list[dict[str, float | int]]:
    if not profile_csv.is_file():
        return FALLBACK_PROFILE_ROWS

    rows: list[dict[str, float | int]] = []
    with profile_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, float | int] = {
                "source_row": int(raw_row["source_row"]),
                "pressure_GPa": float(raw_row["pressure_GPa"]),
                "density_g_cm3": float(raw_row["density_g_cm3"]),
            }
            for column in TEMPERATURE_PROFILES.values():
                row[column] = float(raw_row[column])
            rows.append(row)
    return rows


def selected_conditions(
    profile_rows: list[dict[str, float | int]],
    source_row_start: int,
    source_row_end: int,
) -> list[Condition]:
    if source_row_start > source_row_end:
        raise ValueError("--source-row-start must be <= --source-row-end")

    rows_by_source = {int(row["source_row"]): row for row in profile_rows}
    missing = [
        row
        for row in range(source_row_start, source_row_end + 1)
        if row not in rows_by_source
    ]
    if missing:
        raise ValueError(f"Missing Uranus source_row values: {missing}")

    conditions: list[Condition] = []
    for source_row in range(source_row_start, source_row_end + 1):
        row = rows_by_source[source_row]
        for profile_slug, profile_column in TEMPERATURE_PROFILES.items():
            for composition_slug, composition in COMPOSITIONS.items():
                conditions.append(
                    Condition(
                        source_row=source_row,
                        pressure_gpa=float(row["pressure_GPa"]),
                        density_g_cm3=float(row["density_g_cm3"]),
                        profile_slug=profile_slug,
                        profile_column=profile_column,
                        composition_slug=composition_slug,
                        composition_label=composition["label"],
                        solvent_line=composition["solvent_line"],
                        temperature_k=float(row[profile_column]),
                    )
                )
    return conditions


def replace_line(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
    count: int = 1,
) -> str:
    text, replacements = re.subn(
        pattern,
        replacement,
        text,
        count=count,
        flags=re.MULTILINE,
    )
    if replacements != count:
        raise ValueError(
            f"Expected {count} replacement(s) for {label}, made {replacements}."
        )
    return text


def patch_md_script(
    base_text: str,
    condition: Condition,
    save_interval_steps: int,
    md_steps: int,
    cpus_per_task: int,
) -> str:
    text = base_text

    metadata = "\n".join(
        [
            f'RUN_ID = "{condition.run_id}"',
            f'SOURCE_ROW = {condition.source_row}',
            f'URANUS_PROFILE = "{condition.profile_slug}"',
            f'COMPOSITION = "{condition.composition_label}"',
            f'PRESSURE_GPA = {condition.pressure_gpa:.12g}',
            f'TARGET_DENSITY_G_CM3 = {condition.density_g_cm3:.12g}',
            f'TARGET_TEMPERATURE_K = {condition.temperature_k:.12g}',
            'SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))',
            (
                "PROJECT_ROOT = "
                "os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))"
            ),
            'MD_RESULTS_DIR = os.path.join(SCRIPT_DIR, "MDresults", RUN_ID)',
        ]
    )
    text = replace_line(
        text,
        r"^PROJECT_ROOT\s*=.*\nMD_RESULTS_DIR\s*=.*$",
        metadata,
        "project/output metadata",
    )
    text = replace_line(
        text,
        r'^N_THREADS\s*=\s*["\'].*["\']\s*$',
        f'N_THREADS = "{cpus_per_task}"',
        "thread count",
    )
    text = replace_line(
        text,
        r"^densitygcm3\s*=.*#.*$",
        f"densitygcm3 = {condition.density_g_cm3:.12g} # g/cm^3",
        "density",
    )
    text = replace_line(
        text,
        r"^pressuregpa\s*=.*#.*$",
        f"pressuregpa = {condition.pressure_gpa:.12g} # GPa",
        "pressure",
    )
    text = replace_line(
        text,
        r"^simbox\.add_solvent\(.*$",
        condition.solvent_line,
        "composition",
    )
    text = replace_line(
        text,
        r"^\s*temp\s*=\s*[^,\n]+,",
        f"    temp={condition.temperature_k:.12g},",
        "simpleMD temperature",
    )
    text = replace_line(
        text,
        r'^\s*fname\s*=\s*os\.path\.join\(MD_RESULTS_DIR,.*$',
        '    fname=os.path.join(MD_RESULTS_DIR, f"{RUN_ID}.xyz"),',
        "trajectory filename",
    )
    text = replace_line(
        text,
        r"^\s*s\s*=\s*[^,\n]+,",
        f"    s={save_interval_steps},",
        "save interval",
    )
    text = replace_line(
        text,
        r"^\s*T\s*=\s*[^,\n]+,",
        f"    T={md_steps},",
        "MD steps",
    )
    text = re.sub(r'device\s*=\s*["\']cuda["\']', 'device="cpu"', text)
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
    condition: Condition,
    py_name: str,
    partition: str | None,
    cpus_per_task: int,
) -> str:
    mem_line = find_line(base_slurm_text, "#SBATCH --mem=") or "#SBATCH --mem=64G"
    time_line = find_line(base_slurm_text, "#SBATCH --time=") or "#SBATCH --time=12:00:00"
    ntasks_line = find_line(base_slurm_text, "#SBATCH --ntasks=") or "#SBATCH --ntasks=1"
    source_line = find_line(base_slurm_text, "source ")
    path_line = find_line(base_slurm_text, "export PATH=")
    module_lines = non_cuda_module_lines(base_slurm_text)

    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name=mace_{condition.run_id}",
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
    lines.append("export PYTHONNOUSERSITE=1")
    if path_line:
        lines.append(path_line)
    lines.extend(
        [
            "",
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}",
            "export CUDA_VISIBLE_DEVICES=",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'cd "${SCRIPT_DIR}/../../.."',
            "",
            'echo "Job started on $(hostname)"',
            'echo "Project directory: $(pwd)"',
            'echo "Generated script directory: ${SCRIPT_DIR}"',
            f'echo "Condition: {condition.run_id}"',
            f'echo "Pressure GPa: {condition.pressure_gpa:.12g}"',
            f'echo "Density g/cm^3: {condition.density_g_cm3:.12g}"',
            f'echo "Temperature K: {condition.temperature_k:.12g}"',
            f'echo "Composition: {condition.composition_label}"',
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
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if args.save_interval_steps <= 0:
        raise ValueError("--save-interval-steps must be positive")
    if args.md_steps <= 0:
        raise ValueError("--md-steps must be positive")
    if args.cpus_per_task <= 0:
        raise ValueError("--cpus-per-task must be positive")

    base_py = require_file(args.base_py)
    base_slurm = require_file(args.base_slurm)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_py_text = base_py.read_text(encoding="utf-8")
    base_slurm_text = base_slurm.read_text(encoding="utf-8")
    profile_rows = read_profile_rows(args.profile_csv.resolve())
    conditions = selected_conditions(
        profile_rows,
        source_row_start=args.source_row_start,
        source_row_end=args.source_row_end,
    )

    generated_slurms: list[Path] = []
    for condition in conditions:
        py_name = f"npt_{condition.run_id}.py"
        slurm_name = f"npt_{condition.run_id}.sh"
        py_path = out_dir / py_name
        slurm_path = out_dir / slurm_name

        write_text(
            py_path,
            patch_md_script(
                base_py_text,
                condition=condition,
                save_interval_steps=args.save_interval_steps,
                md_steps=args.md_steps,
                cpus_per_task=args.cpus_per_task,
            ),
            args.overwrite,
        )
        write_text(
            slurm_path,
            patch_slurm_script(
                base_slurm_text,
                condition=condition,
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
        print(
            f"Generated {len(conditions)} Uranus MACE conditions "
            f"({len(generated_slurms)} Slurm scripts). Submit with:"
        )
        print(f"  cd {display_path(out_dir)}")
        print('  for f in *.sh; do sbatch "$f"; done')


if __name__ == "__main__":
    main()
