#!/usr/bin/env python3
"""Generate DAIS ORCA jobs for the focused H2-formation frames.

Run from anywhere. Generated .inp and .slurm files are written directly next
to this script, matching the 7_24b_daisorcalarge workflow.

Input trajectory:
    outputsfull/temperature_ramp/r09_hot_w/plots/focused.xyz

Submit from this folder on DAIS:
    for f in r09_hot_w_h2formation_frame_*.slurm; do sbatch "$f"; done
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
DEFAULT_CLUSTER_XYZ = MLIP_DIR / "outputsfull" / "temperature_ramp" / "r09_hot_w" / "plots" / "focused.xyz"
DEFAULT_CLUSTER_STEM = "r09_hot_w_h2formation_frame"
EXPECTED_CLUSTERS = 20
FAIRCHEM_ORCA_CALC = MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "calc.py"
FAIRCHEM_SRC = MLIP_DIR / "fairchem" / "src"
FAIRCHEM_ORCA_BASIS = MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "basis" / "def2-tzvpd.bas"
LOCAL_BASIS_FALLBACK = MLIP_DIR / "codes" / "7_24b_daisorcalarge" / "def2-tzvpd.bas"
ORCA_SIMPLE_INPUT = (
    "! wB97M-V def2-TZVPD EnGrad RIJCOSX def2/J NoUseSym DIIS NOSOSCF NormalConv DEFGRID3 ALLPOP\n"
    "%pal nprocs 24 end\n"
    '%scf Convergence Tight maxiter 300 end %elprop Dipole true Quadrupole true end '
    '%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 '
    'Print[P_BondOrder_M] 1 Print[P_Fockian] 1 Print[P_OrbEn] 2 end %basis GTOName "def2-tzvpd.bas" end '
    '%scf THRESH 1e-12 TCUT 1e-13 end %maxcore 1000 %nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end \n'
)
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
        return make_input_from_local_template(atoms)

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


def make_input_from_local_template(atoms: Atoms) -> str:
    charge = formal_charge(atom_tuples(atoms))
    lines = [ORCA_SIMPLE_INPUT.rstrip(), f"*xyz {charge} {MULTIPLICITY}"]
    for symbol, x, y, z in atom_tuples(atoms):
        lines.append(f"{symbol}   {x:.10f} {y:.10f} {z:.10f}")
    lines.append("*")
    return "\n".join(lines) + "\n"


def make_slurm(base: str, stem: str) -> str:
    return (
        base.replace("__JOB_NAME__", f"orca_{stem}")
        .replace("__INPUT_FILE__", f"{stem}.inp")
        .replace("__OUTPUT_FILE__", f"{stem}.out")
    )


def cluster_frames(path: Path) -> list[Atoms]:
    if not path.is_file():
        raise FileNotFoundError(f"Multi-frame focused XYZ not found: {path}")
    frames = read(path, ":")
    if len(frames) != EXPECTED_CLUSTERS:
        raise ValueError(f"Expected {EXPECTED_CLUSTERS} frames in {path}, found {len(frames)}")
    return frames


def remove_stale_generated_files() -> int:
    patterns = ("*_h2formation_frame_*.inp", "*_h2formation_frame_*.slurm")
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
    if FAIRCHEM_ORCA_BASIS.exists():
        shutil.copy2(FAIRCHEM_ORCA_BASIS, OUT_DIR / FAIRCHEM_ORCA_BASIS.name)
    elif LOCAL_BASIS_FALLBACK.exists():
        shutil.copy2(LOCAL_BASIS_FALLBACK, OUT_DIR / LOCAL_BASIS_FALLBACK.name)
    else:
        raise FileNotFoundError(
            f"Could not find {FAIRCHEM_ORCA_BASIS.name} at {FAIRCHEM_ORCA_BASIS} "
            f"or fallback {LOCAL_BASIS_FALLBACK}"
        )

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
