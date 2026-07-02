#!/usr/bin/env python3
"""Expand NPTMACEbase.py over Uranus pressure/temperature/composition cases.

Run this file from anywhere.  It writes generated Python and Slurm scripts into the
``expand`` subfolder next to this script.
"""

from __future__ import annotations

import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "NPTMACEbase.py"
BASE_SLURM = SCRIPT_DIR / "NPTMACEbase.slurm"
OUT_DIR = SCRIPT_DIR / "expand"

SAVE_INTERVAL_STEPS = 100
MD_STEPS = 10_000

TEMPERATURES = {
    "hot": "hot_uranus_temperature_K",
    "cold": "cold_uranus_temperature_K",
}

COMPOSITIONS = {
    "w": "simbox.add_solvent([water], ratio=[1], zdim=boxsize, density=densitygcm3)",
    "w4n1": "simbox.add_solvent([water, amm], ratio=[4, 1], zdim=boxsize, density=densitygcm3)",
    "w7n1": "simbox.add_solvent([water, amm], ratio=[7, 1], zdim=boxsize, density=densitygcm3)",
}

PRESSURE_ROWS = [
    {
        "row": 2,
        "pressure_GPa": 0.0001,
        "density_g_cm3": 0.000449,
        "hot_uranus_temperature_K": 76,
        "cold_uranus_temperature_K": 76,
    },
    {
        "row": 3,
        "pressure_GPa": 0.0011,
        "density_g_cm3": 0.00281,
        "hot_uranus_temperature_K": 136,
        "cold_uranus_temperature_K": 156,
    },
    {
        "row": 4,
        "pressure_GPa": 0.01,
        "density_g_cm3": 0.012,
        "hot_uranus_temperature_K": 269,
        "cold_uranus_temperature_K": 269,
    },
    {
        "row": 5,
        "pressure_GPa": 0.11,
        "density_g_cm3": 0.0495,
        "hot_uranus_temperature_K": 537,
        "cold_uranus_temperature_K": 481,
    },
    {
        "row": 6,
        "pressure_GPa": 1,
        "density_g_cm3": 0.14,
        "hot_uranus_temperature_K": 1020,
        "cold_uranus_temperature_K": 854,
    },
    {
        "row": 7,
        "pressure_GPa": 10,
        "density_g_cm3": 0.344,
        "hot_uranus_temperature_K": 2050,
        "cold_uranus_temperature_K": 1500,
    },
    {
        "row": 8,
        "pressure_GPa": 15,
        "density_g_cm3": 0.405,
        "hot_uranus_temperature_K": 2340,
        "cold_uranus_temperature_K": 1640,
    },
    {
        "row": 9,
        "pressure_GPa": 15,
        "density_g_cm3": 1.19,
        "hot_uranus_temperature_K": 2340,
        "cold_uranus_temperature_K": 1640,
    },
    {
        "row": 10,
        "pressure_GPa": 100,
        "density_g_cm3": 3.72,
        "hot_uranus_temperature_K": 5520,
        "cold_uranus_temperature_K": 1920,
    },
]


def replace_once(text: str, pattern: str, replacement: str) -> str:
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not replace pattern: {pattern}")
    return text


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def make_script(base_text: str, row: dict[str, float], temp_name: str, comp_name: str) -> str:
    run_id = f"r{int(row['row']):02d}_{temp_name}_{comp_name}"
    pressure = row["pressure_GPa"]
    density = row["density_g_cm3"]
    temperature = row[TEMPERATURES[temp_name]]

    text = base_text
    text = replace_once(
        text,
        r"^PROJECT_ROOT\s*=.*\nMD_RESULTS_DIR\s*=.*$",
        (
            'SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n'
            'MD_RESULTS_DIR = os.path.join(SCRIPT_DIR, "MDresults", '
            f'"{run_id}")'
        ),
    )
    text = replace_once(
        text,
        r"^densitygcm3\s*=.*$",
        f"densitygcm3 = {density:.12g} # g/cm^3",
    )
    text = replace_once(
        text,
        r"^pressuregpa\s*=.*$",
        f"pressuregpa = {pressure:.12g} # GPa",
    )
    text = replace_once(text, r"^simbox\.add_solvent\(.*$", COMPOSITIONS[comp_name])
    text = replace_once(text, r"^\s*temp\s*=\s*[^,\n]+,", f"    temp={temperature:.12g},")
    text = replace_once(
        text,
        r'^\s*fname\s*=\s*os\.path\.join\(MD_RESULTS_DIR,.*$',
        f'    fname=os.path.join(MD_RESULTS_DIR, "{run_id}.xyz"),',
    )
    text = replace_once(text, r"^\s*s\s*=\s*[^,\n]+,", f"    s={SAVE_INTERVAL_STEPS},")
    text = replace_once(text, r"^\s*T\s*=\s*[^,\n]+,", f"    T={MD_STEPS},")
    return text


def make_slurm(base_text: str, run_id: str, py_name: str) -> str:
    text = base_text
    text = replace_once(text, r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name=npt_{run_id}")
    text = replace_once(text, r"^cd .*$", "cd /ptmp/kshao/mlip/codesVP/6_26_NPT_MACE/expand")
    text = replace_once(
        text,
        r"^srun python .*$",
        (
            "# Local GPU campaign without Slurm: "
            "cd /home/kevinsh/mlip/codesVP/6_26_NPT_MACE && "
            "source ~/env/bin/activate && python NPTMaceexpand.py && cd expand && "
            "CUDA_VISIBLE_DEVICES=0 MLIP_MACE_DEVICE=cuda bash -lc "
            "'for f in npt_*.py; do python \"$f\"; done'\n"
            f"srun python {py_name}"
        ),
    )
    return text


def main() -> None:
    base_text = BASE_SCRIPT.read_text(encoding="utf-8")
    base_slurm_text = BASE_SLURM.read_text(encoding="utf-8")
    OUT_DIR.mkdir(exist_ok=True)

    written_python = 0
    written_slurm = 0
    for row in PRESSURE_ROWS:
        for temp_name in TEMPERATURES:
            for comp_name in COMPOSITIONS:
                run_id = f"r{int(row['row']):02d}_{temp_name}_{comp_name}"
                py_name = f"npt_{run_id}.py"
                slurm_name = f"npt_{run_id}.sh"
                py_path = OUT_DIR / py_name
                slurm_path = OUT_DIR / slurm_name

                write_text_lf(py_path, make_script(base_text, row, temp_name, comp_name))
                write_text_lf(slurm_path, make_slurm(base_slurm_text, run_id, py_name))

                print(f"wrote {py_path.relative_to(SCRIPT_DIR)}")
                print(f"wrote {slurm_path.relative_to(SCRIPT_DIR)}")
                written_python += 1
                written_slurm += 1

    print(
        f"Generated {written_python} Python files and {written_slurm} Slurm files "
        f"in {OUT_DIR.relative_to(SCRIPT_DIR)}"
    )


if __name__ == "__main__":
    main()
