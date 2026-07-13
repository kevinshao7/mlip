#!/usr/bin/env python3
"""Expand ORCA cluster calculations from clusters/*.xyz.

Run from anywhere.  The generated .inp and .slurm files are written to the
expand folder next to this script.

Submit all generated Slurm files from the expand folder:
    cd expand
    for f in *.slurm; do sbatch "$f"; done
"""

from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR / "clusters"
OUT_DIR = SCRIPT_DIR / "expand"
BASE_INP = SCRIPT_DIR / "orcaclustersbase.inp"
BASE_SLURM = SCRIPT_DIR / "orcaclustersbase.slurm"
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
}


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


def format_geometry(atoms: list[tuple[str, float, float, float]]) -> str:
    return "\n".join(f"{symbol:2s} {x: .10f} {y: .10f} {z: .10f}" for symbol, x, y, z in atoms)


def neutral_multiplicity(atoms: list[tuple[str, float, float, float]]) -> int:
    electrons = sum(ATOMIC_NUMBERS[symbol] for symbol, *_ in atoms)
    multiplicity = 2 if electrons % 2 else 1
    if multiplicity not in (1, 2):
        raise ValueError(f"Unsupported multiplicity {multiplicity}; expected 1 or 2")
    return multiplicity


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def make_input(base: str, cluster_xyz: Path) -> str:
    atoms = read_xyz_atoms(cluster_xyz)
    return (
        base.replace("* XYZ 0 1", f"* XYZ 0 {neutral_multiplicity(atoms)}")
        .replace("__GEOMETRY__", format_geometry(atoms))
    )


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

    base_inp = BASE_INP.read_text(encoding="utf-8")
    base_slurm = BASE_SLURM.read_text(encoding="utf-8")
    OUT_DIR.mkdir(exist_ok=True)

    for cluster_xyz in clusters:
        stem = cluster_xyz.stem
        inp_path = OUT_DIR / f"{stem}.inp"
        slurm_path = OUT_DIR / f"{stem}.slurm"
        write_text_lf(inp_path, make_input(base_inp, cluster_xyz))
        write_text_lf(slurm_path, make_slurm(base_slurm, stem))
        print(f"wrote {inp_path.relative_to(SCRIPT_DIR)}")
        print(f"wrote {slurm_path.relative_to(SCRIPT_DIR)}")

    print(f"Generated {len(clusters)} ORCA input files and {len(clusters)} Slurm files in {OUT_DIR.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
