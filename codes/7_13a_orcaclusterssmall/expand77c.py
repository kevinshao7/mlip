#!/usr/bin/env python3
"""Expand ORCA cluster calculations from clusters/*.xyz.

Run from anywhere.  The generated .inp and .slurm files are written to the
expand folder next to this script.

Submit all generated Slurm files from the expand folder:
    cd expand
    for f in *.slurm; do sbatch "$f"; done

PowerShell local sequential ORCA run from the expand folder:
$env:OMP_NUM_THREADS=8; $env:MKL_NUM_THREADS=8; $env:OPENBLAS_NUM_THREADS=8; Get-ChildItem *.inp | Sort-Object Name | ForEach-Object { orca $_.FullName | Tee-Object -FilePath ($_.BaseName + ".out") }
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR / "clusters"
OUT_DIR = SCRIPT_DIR / "expand"
BASE_SLURM = SCRIPT_DIR / "orcaclustersbase.slurm"
MLIP_DIR = SCRIPT_DIR.parents[1]
FAIRCHEM_ORCA_CALC = MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "calc.py"
FAIRCHEM_SRC = MLIP_DIR / "fairchem" / "src"
FAIRCHEM_ORCA_BASIS = MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "basis" / "def2-tzvpd.bas"
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
}
FORMAL_CHARGES = {
    "H": 1,
    "N": -3,
    "O": -2,
}
MULTIPLICITY = 1


def read_xyz_atoms(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file is too short: {path}")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line is not an atom count in {path}") from exc
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"Expected {natoms} atom lines in {path}, found {len(atom_lines)}")

    atoms = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line in {path}: {line}")
        symbol = parts[0]
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"No atomic number configured for {symbol!r} in {path}")
        x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
        atoms.append((symbol, x, y, z))
    return atoms


def formal_charge(atoms: list[tuple[str, float, float, float]]) -> int:
    charge = 0
    for symbol, *_ in atoms:
        try:
            charge += FORMAL_CHARGES[symbol]
        except KeyError as exc:
            raise ValueError(f"No formal charge configured for {symbol!r}") from exc
    return charge


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def load_fairchem_orca_calc():
    if FAIRCHEM_SRC.exists():
        sys.path.insert(0, str(FAIRCHEM_SRC))

    try:
        return importlib.import_module("fairchem.data.omol.orca.calc")
    except ImportError:
        pass

    if not FAIRCHEM_ORCA_CALC.exists():
        return None

    spec = importlib.util.spec_from_file_location("fairchem_omol_orca_calc", FAIRCHEM_ORCA_CALC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load fairchem ORCA calc module from {FAIRCHEM_ORCA_CALC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input_with_fairchem(cluster_xyz: Path, atoms: list[tuple[str, float, float, float]]) -> str:
    orca_calc = load_fairchem_orca_calc()
    if orca_calc is None:
        raise RuntimeError(
            "Could not import fairchem.data.omol.orca.calc and could not load it from "
            f"{FAIRCHEM_ORCA_CALC}"
        )

    charge = formal_charge(atoms)
    from ase.calculators.orca import OrcaProfile
    from ase.io import read

    ase_atoms = read(cluster_xyz)

    def compatible_orca_profile(command):
        if isinstance(command, list):
            command = command[0] or "orca"
        return OrcaProfile(command)

    orca_calc.OrcaProfile = compatible_orca_profile
    OUT_DIR.mkdir(exist_ok=True)
    transient_input = OUT_DIR / "orca.inp"
    orcasimpleinput = orca_calc.ORCA_ASE_SIMPLE_INPUT
    if "PAL8" not in orcasimpleinput.split():
        orcasimpleinput = f"{orcasimpleinput} PAL8"
    orca_calc.write_orca_inputs(
        ase_atoms,
        OUT_DIR,
        charge=charge,
        mult=MULTIPLICITY,
        orcasimpleinput=orcasimpleinput,
    )
    text = transient_input.read_text(encoding="utf-8")
    transient_input.unlink()
    return text


def make_input(cluster_xyz: Path) -> str:
    atoms = read_xyz_atoms(cluster_xyz)
    return make_input_with_fairchem(cluster_xyz, atoms)


def make_slurm(base: str, stem: str) -> str:
    return (
        base.replace("__JOB_NAME__", f"orca_{stem[:26]}")
        .replace("__INPUT_FILE__", f"{stem}.inp")
        .replace("__OUTPUT_FILE__", f"{stem}.out")
    )


def main() -> None:
    clusters = sorted(CLUSTER_DIR.glob("*.xyz"))
    if not clusters:
        raise FileNotFoundError(f"No cluster .xyz files found in {CLUSTER_DIR}")

    base_slurm = BASE_SLURM.read_text(encoding="utf-8")
    OUT_DIR.mkdir(exist_ok=True)
    if FAIRCHEM_ORCA_BASIS.exists():
        shutil.copy2(FAIRCHEM_ORCA_BASIS, OUT_DIR / FAIRCHEM_ORCA_BASIS.name)

    for cluster_xyz in clusters:
        stem = cluster_xyz.stem
        inp_path = OUT_DIR / f"{stem}.inp"
        slurm_path = OUT_DIR / f"{stem}.slurm"
        write_text_lf(inp_path, make_input(cluster_xyz))
        write_text_lf(slurm_path, make_slurm(base_slurm, stem))
        print(f"wrote {inp_path.relative_to(SCRIPT_DIR)}")
        print(f"wrote {slurm_path.relative_to(SCRIPT_DIR)}")

    print(f"Generated {len(clusters)} ORCA input files and {len(clusters)} Slurm files in {OUT_DIR.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
