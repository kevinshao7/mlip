#!/usr/bin/env python3
"""Generate the 20 DAIS production NPT jobs for selected Uranus conditions."""

from __future__ import annotations

import csv
import re
from fractions import Fraction
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "NPTMACEproduction_base.py"
BASE_SLURM = SCRIPT_DIR / "NPTMACEproduction_base.slurm"
PROFILE_CSV = SCRIPT_DIR.parent / "old" / "7_6b_uranusprofile" / "uranus_profiles.csv"
OUT_DIR = SCRIPT_DIR / "expand"

SAVE_INTERVAL_STEPS = 5
# 2 ns at the base script timestep of 0.5 fs/step.
MD_STEPS: int | None = 4_000_000

PRESSURES_GPA = [100.0, 10.0, 1.0, 0.11]
TEMPERATURE_COLUMN = "preferred_uranus_temperature_K"

# NH3/H2O molar ratios.  Values become exact integer ratios for mdinterface.
AMMONIA_WATER_RATIOS = [0.0, 0.1, 0.2, 0.5, 1.0]

WATER_MOLAR_MASS = 18.01528
AMMONIA_MOLAR_MASS = 17.03052


def replace_once(text: str, pattern: str, replacement: str) -> str:
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not replace pattern: {pattern}")
    return text


def replace_assignment(text: str, name: str, replacement: str) -> str:
    return replace_once(text, rf"^{name}\s*=.*$", replacement)


def replace_call_keyword(text: str, name: str, replacement_value: str) -> str:
    return replace_once(text, rf"^(\s*){name}\s*=\s*[^,\n]+,", rf"\1{name}={replacement_value},")


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def pressure_tag(pressure_gpa: float) -> str:
    return f"P{pressure_gpa:g}GPa".replace(".", "p")


def ratio_tag(ratio: float) -> str:
    return f"R{ratio:g}".replace(".", "p")


def ratio_expression(ratio: float) -> str:
    if ratio == 0:
        return "simbox.add_solvent([water], ratio=[1], zdim=boxsize, density=densitygcm3)"

    frac = Fraction(str(ratio)).limit_denominator()
    water_count = frac.denominator
    ammonia_count = frac.numerator
    return (
        "simbox.add_solvent([water, amm], "
        f"ratio=[{water_count}, {ammonia_count}], "
        "zdim=boxsize, density=densitygcm3)"
    )


def mixture_molar_mass(ratio: float) -> float:
    if ratio == 0:
        return WATER_MOLAR_MASS
    frac = Fraction(str(ratio)).limit_denominator()
    water_count = frac.denominator
    ammonia_count = frac.numerator
    return (
        water_count * WATER_MOLAR_MASS + ammonia_count * AMMONIA_MOLAR_MASS
    ) / (water_count + ammonia_count)


def load_profile_rows() -> dict[float, dict[str, float]]:
    rows: dict[float, dict[str, float]] = {}
    with PROFILE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pressure = float(row["pressure_GPa"])
            rows[pressure] = {
                "source_row": float(row["source_row"]),
                "pressure_GPa": pressure,
                "density_g_cm3": float(row["density_g_cm3"]),
                "temperature_K": float(row[TEMPERATURE_COLUMN]),
            }

    missing = [pressure for pressure in PRESSURES_GPA if pressure not in rows]
    if missing:
        raise RuntimeError(f"Missing requested pressure rows in {PROFILE_CSV}: {missing}")
    return rows


def make_script(base_text: str, row: dict[str, float], ratio: float) -> str:
    pressure = row["pressure_GPa"]
    temperature = row["temperature_K"]
    ptag = pressure_tag(pressure)
    rtag = ratio_tag(ratio)
    run_id = f"{ptag}_{rtag}"

    text = base_text
    text = replace_assignment(
        text,
        "PROJECT_ROOT",
        "PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))",
    )
    text = replace_assignment(
        text,
        "MD_RESULTS_DIR",
        f'MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputsfull", "conditionsproduction", "{run_id}")',
    )
    text = replace_assignment(text, "densitygcm3", "densitygcm3 = 0.2 # initial build density, g/cm3")
    text = replace_assignment(text, "target_profile_densitygcm3", f"target_profile_densitygcm3 = {row['density_g_cm3']:.12g} # Uranus profile source row {int(row['source_row'])}, g/cm3")
    text = replace_assignment(text, "pressuregpa", f"pressuregpa = {pressure:.12g} # GPa")
    text = replace_assignment(text, "moleculemass", f"moleculemass = {mixture_molar_mass(ratio):.12g} # grams per mol, composition-weighted")
    text = replace_assignment(text, "T_final", f"T_final = {temperature:.12g}  # {TEMPERATURE_COLUMN}")
    text = replace_assignment(text, "composition_label", f'composition_label = "{rtag}"')
    text = replace_assignment(text, "ammonia_water_ratio", f"ammonia_water_ratio = {ratio:.12g} # NH3/H2O molar ratio")
    text = replace_once(text, r"^simbox\.add_solvent\(.*$", ratio_expression(ratio))
    text = replace_call_keyword(text, "s", str(SAVE_INTERVAL_STEPS))
    if MD_STEPS is not None:
        text = replace_assignment(text, "totaltimesteps", f"totaltimesteps = {MD_STEPS}  # 2 ns at 0.5 fs/step")
    return text


def make_slurm(base_text: str, run_id: str, py_name: str) -> str:
    text = base_text
    text = replace_once(text, r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name=cond_{run_id}")
    text = replace_once(text, r"^#SBATCH --chdir=.*$", "#SBATCH --chdir=/dais/fs/scratch/kshao/mlip/codes/B_conditionsproduction/expand")
    text = replace_assignment(text, "PYTHON_SCRIPT", f"PYTHON_SCRIPT={py_name}")
    return text


def main() -> None:
    base_text = BASE_SCRIPT.read_text(encoding="utf-8")
    base_slurm_text = BASE_SLURM.read_text(encoding="utf-8")
    rows = load_profile_rows()
    OUT_DIR.mkdir(exist_ok=True)

    written_python = 0
    written_slurm = 0
    manifest_lines = [
        "run_id,pressure_GPa,density_g_cm3,temperature_K,ammonia_water_ratio,python,slurm"
    ]

    for pressure in PRESSURES_GPA:
        row = rows[pressure]
        for ratio in AMMONIA_WATER_RATIOS:
            run_id = f"{pressure_tag(pressure)}_{ratio_tag(ratio)}"
            py_name = f"production_{run_id}.py"
            slurm_name = f"production_{run_id}.sh"
            py_path = OUT_DIR / py_name
            slurm_path = OUT_DIR / slurm_name

            write_text_lf(py_path, make_script(base_text, row, ratio))
            write_text_lf(slurm_path, make_slurm(base_slurm_text, run_id, py_name))

            manifest_lines.append(
                f"{run_id},{pressure:g},{row['density_g_cm3']:g},{row['temperature_K']:g},{ratio:g},"
                f"expand/{py_name},expand/{slurm_name}"
            )
            print(f"wrote {py_path.relative_to(SCRIPT_DIR)}")
            print(f"wrote {slurm_path.relative_to(SCRIPT_DIR)}")
            written_python += 1
            written_slurm += 1

    write_text_lf(SCRIPT_DIR / "production_manifest.csv", "\n".join(manifest_lines) + "\n")
    print(
        f"Generated {written_python} Python files and {written_slurm} Slurm files "
        f"in {OUT_DIR.relative_to(SCRIPT_DIR)}"
    )


if __name__ == "__main__":
    main()
