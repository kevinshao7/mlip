#!/usr/bin/env python3
"""Generate fine-grid H2O-NH3 NPT jobs between 10 and 100 GPa."""

from __future__ import annotations

import csv
import math
import re
from fractions import Fraction
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR.parent / "B_conditionsproduction"
BASE_SCRIPT = REFERENCE_DIR / "NPTMACEproduction_base.py"
BASE_SLURM = REFERENCE_DIR / "NPTMACEproduction_base.slurm"
PROFILE_CSV = SCRIPT_DIR.parent / "old" / "7_6b_uranusprofile" / "uranus_profiles.csv"
OUT_DIR = SCRIPT_DIR / "expand"

SAVE_INTERVAL_STEPS = 5
MD_STEPS = 8_000_000  # 4 ns at 0.5 fs/step
PRESSURES_GPA = [10.0 ** 1.25, 10.0 ** 1.5, 10.0 ** 1.75]
AMMONIA_WATER_RATIOS = [0.0, 0.1, 0.2, 0.5, 1.0]
TEMPERATURE_COLUMN = "preferred_uranus_temperature_K"
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
    return f"P{pressure_gpa:.4f}GPa".replace(".", "p")


def ratio_tag(ratio: float) -> str:
    return f"R{ratio:g}".replace(".", "p")


def ratio_expression(ratio: float) -> str:
    if ratio == 0:
        return "simbox.add_solvent([water], ratio=[1], zdim=boxsize, density=densitygcm3)"
    frac = Fraction(str(ratio)).limit_denominator()
    return (
        "simbox.add_solvent([water, amm], "
        f"ratio=[{frac.denominator}, {frac.numerator}], "
        "zdim=boxsize, density=densitygcm3)"
    )


def mixture_molar_mass(ratio: float) -> float:
    frac = Fraction(str(ratio)).limit_denominator()
    return (
        frac.denominator * WATER_MOLAR_MASS + frac.numerator * AMMONIA_MOLAR_MASS
    ) / (frac.denominator + frac.numerator)


def load_endpoint_rows() -> dict[float, dict[str, float]]:
    rows: dict[float, dict[str, float]] = {}
    with PROFILE_CSV.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            pressure = float(raw["pressure_GPa"])
            if pressure in (10.0, 100.0):
                rows[pressure] = {
                    "density_g_cm3": float(raw["density_g_cm3"]),
                    "temperature_K": float(raw[TEMPERATURE_COLUMN]),
                }
    if set(rows) != {10.0, 100.0}:
        raise RuntimeError(f"Could not find unique 10 and 100 GPa endpoints in {PROFILE_CSV}")
    return rows


def interpolate_condition(pressure: float, endpoints: dict[float, dict[str, float]]) -> dict[str, float]:
    # Linear interpolation in log10(P), as specified for this fine pressure grid.
    fraction = (math.log10(pressure) - 1.0) / (2.0 - 1.0)
    low, high = endpoints[10.0], endpoints[100.0]
    return {
        "pressure_GPa": pressure,
        "density_g_cm3": low["density_g_cm3"] + fraction * (high["density_g_cm3"] - low["density_g_cm3"]),
        "temperature_K": low["temperature_K"] + fraction * (high["temperature_K"] - low["temperature_K"]),
        "log_pressure_fraction": fraction,
    }


def make_script(base_text: str, row: dict[str, float], ratio: float) -> str:
    pressure = row["pressure_GPa"]
    ptag, rtag = pressure_tag(pressure), ratio_tag(ratio)
    run_id = f"{ptag}_{rtag}"
    text = base_text
    text = replace_assignment(text, "PROJECT_ROOT", "PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))")
    text = replace_assignment(text, "MD_RESULTS_DIR", f'MD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputsfull", "conditionsfine", "{run_id}")')
    text = replace_assignment(text, "densitygcm3", "densitygcm3 = 0.2 # initial build density, g/cm3")
    text = replace_assignment(text, "target_profile_densitygcm3", f"target_profile_densitygcm3 = {row['density_g_cm3']:.12g} # log-pressure-interpolated profile density, g/cm3")
    text = replace_assignment(text, "pressuregpa", f"pressuregpa = {pressure:.12g} # GPa")
    text = replace_assignment(text, "moleculemass", f"moleculemass = {mixture_molar_mass(ratio):.12g} # grams per mol, composition-weighted")
    text = replace_assignment(text, "T_final", f"T_final = {row['temperature_K']:.12g}  # log-pressure-interpolated {TEMPERATURE_COLUMN}")
    text = replace_assignment(text, "composition_label", f'composition_label = "{rtag}"')
    text = replace_assignment(text, "ammonia_water_ratio", f"ammonia_water_ratio = {ratio:.12g} # NH3/H2O molar ratio")
    text = replace_once(text, r"^simbox\.add_solvent\(.*$", ratio_expression(ratio))
    text = replace_call_keyword(text, "s", str(SAVE_INTERVAL_STEPS))
    text = replace_assignment(text, "totaltimesteps", f"totaltimesteps = {MD_STEPS}  # 4 ns at 0.5 fs/step")
    return text


def make_slurm(base_text: str, run_id: str, py_name: str) -> str:
    text = replace_once(base_text, r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name=fine_{run_id}")
    text = replace_once(text, r"^#SBATCH --chdir=.*$", "#SBATCH --chdir=/dais/fs/scratch/kshao/mlip/codes/B2_conditionsfine/expand")
    return replace_assignment(text, "PYTHON_SCRIPT", f"PYTHON_SCRIPT={py_name}")


def main() -> None:
    base_text = BASE_SCRIPT.read_text(encoding="utf-8")
    base_slurm = BASE_SLURM.read_text(encoding="utf-8")
    endpoints = load_endpoint_rows()
    OUT_DIR.mkdir(exist_ok=True)
    manifest = ["run_id,pressure_GPa,density_g_cm3,temperature_K,log_pressure_fraction,ammonia_water_ratio,python,slurm"]
    for pressure in PRESSURES_GPA:
        row = interpolate_condition(pressure, endpoints)
        for ratio in AMMONIA_WATER_RATIOS:
            run_id = f"{pressure_tag(pressure)}_{ratio_tag(ratio)}"
            py_name, slurm_name = f"production_{run_id}.py", f"production_{run_id}.sh"
            write_text_lf(OUT_DIR / py_name, make_script(base_text, row, ratio))
            write_text_lf(OUT_DIR / slurm_name, make_slurm(base_slurm, run_id, py_name))
            manifest.append(f"{run_id},{pressure:.12g},{row['density_g_cm3']:.12g},{row['temperature_K']:.12g},{row['log_pressure_fraction']:.2f},{ratio:g},expand/{py_name},expand/{slurm_name}")
    write_text_lf(SCRIPT_DIR / "production_manifest.csv", "\n".join(manifest) + "\n")
    print(f"Generated {len(PRESSURES_GPA) * len(AMMONIA_WATER_RATIOS)} Python/Slurm job pairs in {OUT_DIR}")


if __name__ == "__main__":
    main()
