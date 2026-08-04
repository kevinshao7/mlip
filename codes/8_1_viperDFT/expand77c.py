#!/usr/bin/env python3
"""Generate ORCA jobs for the first 100 H2-formation training clusters.

Run from anywhere. Generated .inp and .slurm files are written directly next
to this script because expand/ is ignored by .gitignore.

Submit from this folder:
    for f in r09_hot_w_h2training_first100_*.slurm; do sbatch "$f"; done
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

from ase import Atoms
from ase.io import read


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
BASE_SLURM = SCRIPT_DIR / "orcaclustersbase.slurm"
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CLUSTER_XYZ = (
    MLIP_DIR
    / "outputsfull"
    / "7_26_H2pathvalidation"
    / "r09_hot_w_h2formation_training_clusters.xyz"
)
DEFAULT_CLUSTER_STEM = "r09_hot_w_h2training_first100"
FRAME_START = 0
EXPECTED_CLUSTERS = 100
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
    "S": -2,
}
MULTIPLICITY = 1


def atom_tuples(atoms: Atoms) -> list[tuple[str, float, float, float]]:
    rows = []
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"No atomic number configured for {symbol!r}")
        rows.append((symbol, float(position[0]), float(position[1]), float(position[2])))
    return rows


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


def copy_basis_to_output(source: Path) -> None:
    destination = OUT_DIR / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


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


def make_input_with_fairchem(atoms: Atoms) -> str:
    orca_calc = load_fairchem_orca_calc()
    if orca_calc is None:
        raise RuntimeError(
            "Could not import fairchem.data.omol.orca.calc and could not load it from "
            f"{FAIRCHEM_ORCA_CALC}"
        )

    charge = formal_charge(atom_tuples(atoms))
    from ase.calculators.orca import OrcaProfile

    def compatible_orca_profile(command):
        if isinstance(command, list):
            command = command[0] or "orca"
        return OrcaProfile(command)

    orca_calc.OrcaProfile = compatible_orca_profile
    transient_input = OUT_DIR / "orca.inp"
    orcasimpleinput = orca_calc.ORCA_ASE_SIMPLE_INPUT
    orcasimpleinput = " ".join(
        token
        for token in orcasimpleinput.split()
        if not (token.upper().startswith("PAL") and token[3:].isdigit())
    )
    orca_calc.write_orca_inputs(
        atoms,
        OUT_DIR,
        charge=charge,
        mult=MULTIPLICITY,
        orcasimpleinput=orcasimpleinput,
    )
    text = transient_input.read_text(encoding="utf-8")
    transient_input.unlink()
    lines = text.splitlines()
    if lines and not any(line.strip().lower().startswith("%pal") for line in lines):
        lines.insert(1, "%pal nprocs 24 end")
        text = "\n".join(lines) + "\n"
    return text


def make_slurm(base: str, stem: str) -> str:
    return (
        base.replace("__JOB_NAME__", f"orca_{stem}")
        .replace("__INPUT_FILE__", f"{stem}.inp")
        .replace("__OUTPUT_FILE__", f"{stem}.out")
    )


def cluster_frames(path: Path) -> list[Atoms]:
    if not path.is_file():
        raise FileNotFoundError(f"Multi-frame cluster XYZ not found: {path}")
    frames = read(path, ":")
    frame_stop = FRAME_START + EXPECTED_CLUSTERS
    if len(frames) < frame_stop:
        raise ValueError(
            f"Expected at least {frame_stop} cluster frames in {path}, found {len(frames)}"
        )
    return frames[FRAME_START:frame_stop]


def remove_stale_generated_files() -> int:
    patterns = ("r09_hot_w_h2training_first100_*.inp", "r09_hot_w_h2training_first100_*.slurm")
    stale_files = [path for pattern in patterns for path in OUT_DIR.glob(pattern)]
    for path in stale_files:
        path.unlink()
    return len(stale_files)


def main() -> None:
    clusters = cluster_frames(DEFAULT_CLUSTER_XYZ)
    base_slurm = BASE_SLURM.read_text(encoding="utf-8")
    stale_count = remove_stale_generated_files()
    if stale_count:
        print(f"Removed {stale_count} stale generated input/slurm files from {OUT_DIR}")
    if not FAIRCHEM_ORCA_BASIS.exists():
        raise FileNotFoundError(f"Could not find required fairchem ORCA basis file: {FAIRCHEM_ORCA_BASIS}")
    copy_basis_to_output(FAIRCHEM_ORCA_BASIS)

    for cluster_index, atoms in enumerate(clusters, start=1):
        stem = f"{DEFAULT_CLUSTER_STEM}_{cluster_index:03d}"
        inp_path = OUT_DIR / f"{stem}.inp"
        slurm_path = OUT_DIR / f"{stem}.slurm"
        write_text_lf(inp_path, make_input_with_fairchem(atoms))
        write_text_lf(slurm_path, make_slurm(base_slurm, stem))
        print(f"wrote {inp_path.name}")
        print(f"wrote {slurm_path.name}")

    print(f"Generated {len(clusters)} ORCA input files and {len(clusters)} Slurm files in {OUT_DIR}")


if __name__ == "__main__":
    main()
