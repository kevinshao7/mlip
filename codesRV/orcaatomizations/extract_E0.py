#!/usr/bin/env python3
"""Extract isolated-atom E0 values from ORCA output files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HARTREE_TO_EV = 27.211386245988
ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def parse_energy_hartree(path: Path) -> float:
    energy_hartree: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ENERGY_RE.search(line)
        if match:
            energy_hartree = float(match.group(1))
    if energy_hartree is None:
        raise ValueError(f"No FINAL SINGLE POINT ENERGY found in {path}")
    return energy_hartree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ORCA isolated atom reference energies."
    )
    parser.add_argument(
        "outputs",
        nargs="*",
        type=Path,
        default=sorted(Path(".").glob("orcaatomization*.out")),
        help="ORCA output files. Defaults to orcaatomization*.out in cwd.",
    )
    return parser.parse_args()


def atom_from_name(path: Path) -> str:
    stem = path.stem
    prefix = "orcaatomization"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def main() -> None:
    args = parse_args()
    if not args.outputs:
        raise SystemExit("No ORCA output files found.")

    print("# atom E0_hartree E0_eV")
    for path in sorted(args.outputs):
        energy_hartree = parse_energy_hartree(path)
        print(f"{atom_from_name(path):2s} {energy_hartree: .12f} {energy_hartree * HARTREE_TO_EV: .12f}")


if __name__ == "__main__":
    main()
