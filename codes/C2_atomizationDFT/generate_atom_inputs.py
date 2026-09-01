#!/usr/bin/env python3
"""Generate isolated-atom ORCA inputs with FairChem's ORCA writer."""

from __future__ import annotations

import importlib
import shutil
import re
from pathlib import Path

from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
FAIRCHEM_SRC = MLIP_DIR / "fairchem" / "src"
FAIRCHEM_ORCA_BASIS = (
    FAIRCHEM_SRC / "fairchem" / "data" / "omol" / "orca" / "basis" / "def2-tzvpd.bas"
)
LOCAL_BASIS = SCRIPT_DIR / "def2-tzvpd.bas"
RUNS_DIR = SCRIPT_DIR / "runs"
MANIFEST_PATH = SCRIPT_DIR / "manifest.csv"
ORCA_PAL_NPROCS = 16

ATOM_SPECS = [
    {"atom": "H", "charge": 0, "multiplicity": 2},
    {"atom": "O", "charge": 0, "multiplicity": 3},
    {"atom": "N", "charge": 0, "multiplicity": 4},
    {"atom": "S", "charge": 0, "multiplicity": 3},
]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_fairchem_orca_calc():
    try:
        return importlib.import_module("fairchem.data.omol.orca.calc")
    except ImportError as exc:
        fail(
            "FairChem is required to generate ORCA inputs. Could not import "
            "fairchem.data.omol.orca.calc; install/activate FairChem before running this script."
        )
        raise exc


def ensure_orca_parallel_settings(text: str) -> str:
    pal_line = f"%pal nprocs {ORCA_PAL_NPROCS} end"
    lines = text.splitlines()
    filtered_lines: list[str] = []
    skip_pal_block = False
    for line in lines:
        stripped = line.strip()
        if skip_pal_block:
            if stripped.lower() == "end":
                skip_pal_block = False
            continue
        if re.match(r"(?i)^%pal\b", stripped):
            if stripped.lower() != "end" and not re.search(r"(?i)\bend\b", stripped):
                skip_pal_block = True
            continue
        filtered_lines.append(line)

    insert_at = 1 if filtered_lines and filtered_lines[0].lstrip().startswith("!") else 0
    filtered_lines.insert(insert_at, pal_line)
    return "\n".join(filtered_lines) + "\n"


def validate_fairchem_input(text: str, charge: int, multiplicity: int, atom: str) -> None:
    if re.search(r"{{[A-Z_]+}}", text):
        fail(f"FairChem-generated ORCA input for {atom} contains template placeholders")
    if "%loc" in text:
        fail(f"FairChem-generated ORCA input for {atom} unexpectedly contains a %loc block")
    stripped_lines = [line.strip() for line in text.splitlines()]
    required_fragments = [
        "! wB97M-V def2-TZVPD",
        f"%pal nprocs {ORCA_PAL_NPROCS} end",
        "EnGrad",
        "RIJCOSX",
        'GTOName "def2-tzvpd.bas"',
        '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end',
        f"*xyz {charge} {multiplicity}",
        f"{atom}   0.0 0.0 0.0",
    ]
    for fragment in required_fragments:
        if fragment not in text:
            fail(f"FairChem-generated ORCA input for {atom} is missing expected fragment: {fragment}")
    if not stripped_lines or stripped_lines[-1] != "*":
        fail(f"FairChem-generated ORCA input for {atom} must end with '*'")


def make_input_with_fairchem(orca_calc, run_dir: Path, atom: str, charge: int, multiplicity: int) -> str:
    from ase.calculators.orca import OrcaProfile

    def compatible_orca_profile(command):
        if isinstance(command, list):
            command = command[0] or "orca"
        return OrcaProfile(command)

    orca_calc.OrcaProfile = compatible_orca_profile
    run_dir.mkdir(parents=True, exist_ok=True)
    transient_input = run_dir / "orca.inp"
    if transient_input.exists():
        transient_input.unlink()
    orca_calc.write_orca_inputs(
        Atoms(symbols=[atom], positions=[(0.0, 0.0, 0.0)]),
        run_dir,
        charge=charge,
        mult=multiplicity,
    )
    if not transient_input.is_file():
        fail(f"FairChem did not write expected transient ORCA input: {transient_input}")
    text = ensure_orca_parallel_settings(transient_input.read_text(encoding="utf-8"))
    transient_input.unlink()
    validate_fairchem_input(text, charge, multiplicity, atom)
    return text


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    orca_calc = load_fairchem_orca_calc()
    if not FAIRCHEM_ORCA_BASIS.is_file():
        fail(f"Missing FairChem ORCA basis file: {FAIRCHEM_ORCA_BASIS}")

    shutil.copy2(FAIRCHEM_ORCA_BASIS, LOCAL_BASIS)

    manifest_lines = ["atom,charge,multiplicity,run_dir,input_path,output_path"]
    for spec in ATOM_SPECS:
        atom = str(spec["atom"])
        charge = int(spec["charge"])
        multiplicity = int(spec["multiplicity"])
        stem = f"orcaatomization{atom}"
        run_dir = RUNS_DIR / atom
        inp_path = run_dir / f"{stem}.inp"
        out_path = run_dir / f"{stem}.out"
        write_text_lf(inp_path, make_input_with_fairchem(orca_calc, run_dir, atom, charge, multiplicity))
        shutil.copy2(LOCAL_BASIS, run_dir / LOCAL_BASIS.name)
        manifest_lines.append(
            f"{atom},{charge},{multiplicity},{run_dir.relative_to(SCRIPT_DIR)},"
            f"{inp_path.relative_to(SCRIPT_DIR)},{out_path.relative_to(SCRIPT_DIR)}"
        )
        print(f"wrote {inp_path.relative_to(SCRIPT_DIR)}")

    write_text_lf(MANIFEST_PATH, "\n".join(manifest_lines) + "\n")
    print(f"wrote {MANIFEST_PATH.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
