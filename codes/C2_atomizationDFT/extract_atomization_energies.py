#!/usr/bin/env python3
"""Extract isolated-atom ORCA energies and rewrite atomizationenergies.txt."""

from __future__ import annotations

import csv
import re
from pathlib import Path


HARTREE_TO_EV = 27.211386245988
ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
NORMAL_TERMINATION = "ORCA TERMINATED NORMALLY"
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_PATH = SCRIPT_DIR.parent / "7_7b_clustervalidation" / "atomizationenergies.txt"
RUNS_DIR = SCRIPT_DIR / "runs"
ATOM_ORDER = ["H", "O", "N", "S"]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def parse_energy_hartree(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    if NORMAL_TERMINATION not in text:
        fail(f"ORCA output did not terminate normally: {path}")
    matches = ENERGY_RE.findall(text)
    if not matches:
        fail(f"No FINAL SINGLE POINT ENERGY found in {path}")
    return float(matches[-1])


def write_target(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Atom", " DFT Atomization energies (eV) to omegaB97 standard"])
        for atom, energy_ev in rows:
            writer.writerow([atom, f"{energy_ev:.5f}"])


def main() -> None:
    rows: list[tuple[str, float]] = []
    for atom in ATOM_ORDER:
        out_path = RUNS_DIR / atom / f"orcaatomization{atom}.out"
        if not out_path.is_file():
            fail(f"Missing ORCA output: {out_path}")
        energy_ev = parse_energy_hartree(out_path) * HARTREE_TO_EV
        rows.append((atom, energy_ev))
        print(f"{atom} {energy_ev:.8f} eV")

    write_target(TARGET_PATH, rows)
    print(f"wrote {TARGET_PATH}")


if __name__ == "__main__":
    main()
