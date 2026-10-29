#!/usr/bin/env python3
"""Prepare, but do not run, ORCA inputs for the four C4 pilot clusters."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from ase.data import atomic_numbers
from ase.io import read


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CLUSTERS = SCRIPT_DIR / "clusters_20GPa_pilot4.xyz"
DEFAULT_OUTPUT = SCRIPT_DIR / "expand"
TEMPLATE = SCRIPT_DIR / "base.inp"
SOURCE_BASIS = MLIP_DIR / "codes" / "C3_DFTproductionstopH2" / "def2-tzvpd.bas"
FORMAL_CHARGES = {"H": 1, "N": -3, "O": -2, "S": -2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def formal_charge(symbols: list[str]) -> int:
    missing = sorted(set(symbols) - FORMAL_CHARGES.keys())
    if missing:
        raise ValueError(f"No formal-charge rule for elements: {', '.join(missing)}")
    return sum(FORMAL_CHARGES[symbol] for symbol in symbols)


def multiplicity(symbols: list[str], charge: int) -> int:
    electrons = sum(int(atomic_numbers[symbol]) for symbol in symbols) - charge
    return 2 if electrons % 2 else 1


def main() -> None:
    args = parse_args()
    frames = list(read(args.clusters, index=":"))
    if len(frames) != 4:
        raise ValueError(f"Pilot cluster file must contain exactly four frames, found {len(frames)}")
    template = TEMPLATE.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not SOURCE_BASIS.is_file():
        raise FileNotFoundError(f"Required ORCA basis file is missing: {SOURCE_BASIS}")
    shutil.copy2(SOURCE_BASIS, args.output_dir / SOURCE_BASIS.name)

    for index, atoms in enumerate(frames, start=1):
        symbols = atoms.get_chemical_symbols()
        charge = formal_charge(symbols)
        spin = multiplicity(symbols, charge)
        stored_charge, stored_spin = int(atoms.info["charge"]), int(atoms.info["spin"])
        if (charge, spin) != (stored_charge, stored_spin):
            raise ValueError(
                f"Cluster {index}: recomputed charge/spin {(charge, spin)} disagree with "
                f"stored values {(stored_charge, stored_spin)}"
            )
        coordinates = "\n".join(
            f"{symbol:<2s} {x: .12f} {y: .12f} {z: .12f}"
            for symbol, (x, y, z) in zip(symbols, atoms.positions)
        )
        text = template.replace("{{CHARGE}}", str(charge)).replace("{{MULTIPLICITY}}", str(spin))
        text = text.replace("{{COORDINATES}}", coordinates)
        if re.search(r"{{[A-Z_]+}}", text):
            raise RuntimeError("Unresolved ORCA template placeholder")
        destination = args.output_dir / f"C4_fine20GPa_cluster_{index:04d}.inp"
        if destination.exists() and not args.force:
            raise FileExistsError(f"Input exists; pass --force to replace: {destination}")
        destination.write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {destination} (charge={charge}, multiplicity={spin}, atoms={len(atoms)})")


if __name__ == "__main__":
    main()
