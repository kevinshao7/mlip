#!/usr/bin/env python3
"""Extract completed D4rev isolated-atom ORCA energies in eV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from generate_atom_inputs import ATOM_SPECS, FUNCTIONAL_BASIS, RUNS_DIR, SCRIPT_DIR, check_inputs


HARTREE_TO_EV = 27.211386245988
ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
NORMAL_TERMINATION = "ORCA TERMINATED NORMALLY"
DEFAULT_OUTPUT = SCRIPT_DIR / "atomizationenergies_D4rev.txt"
DEFAULT_JSON_OUTPUT = SCRIPT_DIR / "e0s_wb97m_d4rev.json"
ATOMIC_NUMBERS = {"H": 1, "N": 7, "O": 8, "S": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing extracted reference-energy file.",
    )
    return parser.parse_args()


def parse_energy_hartree(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ORCA output: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if NORMAL_TERMINATION not in text:
        raise ValueError(f"ORCA output did not terminate normally: {path}")
    matches = ENERGY_RE.findall(text)
    if not matches:
        raise ValueError(f"No FINAL SINGLE POINT ENERGY found in {path}")
    return float(matches[-1])


def main() -> None:
    args = parse_args()
    check_inputs()
    existing = [path for path in (args.output, args.json_output) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Output exists; pass --force to replace: {existing[0]}")

    rows: list[tuple[str, float]] = []
    for spec in ATOM_SPECS:
        stem = f"orcaatomizationD4rev{spec.symbol}"
        energy_ev = parse_energy_hartree(RUNS_DIR / spec.symbol / f"{stem}.out") * HARTREE_TO_EV
        rows.append((spec.symbol, energy_ev))
        print(f"{spec.symbol} {energy_ev:.10f} eV")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Atom", f" DFT atomic reference energy (eV), {FUNCTIONAL_BASIS}"])
        for atom, energy_ev in rows:
            writer.writerow([atom, f"{energy_ev:.8f}"])
    print(f"wrote {args.output}")

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    e0s = {
        str(ATOMIC_NUMBERS[species.symbol]): float(f"{energy:.8f}")
        for species, (_, energy) in zip(ATOM_SPECS, rows, strict=True)
    }
    args.json_output.write_text(
        json.dumps(dict(sorted(e0s.items(), key=lambda item: int(item[0]))), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
