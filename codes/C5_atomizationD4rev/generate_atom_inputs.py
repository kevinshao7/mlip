#!/usr/bin/env python3
"""Generate and validate neutral-atom ORCA inputs for D4rev references."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
MANIFEST_PATH = SCRIPT_DIR / "manifest.csv"
FUNCTIONAL_BASIS = "wB97M-D4rev def2-TZVPD"
ORCA_NPROCS = 20


@dataclass(frozen=True)
class AtomSpec:
    symbol: str
    charge: int
    multiplicity: int


# These are neutral, isolated atomic reference states, not molecular fragments.
# Their formal charge is therefore explicitly zero. Multiplicities are the
# ground-state values: H 2S, N 4S, O 3P, and S 3P.
ATOM_SPECS = (
    AtomSpec("H", 0, 2),
    AtomSpec("O", 0, 3),
    AtomSpec("N", 0, 4),
    AtomSpec("S", 0, 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing inputs and manifest without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing inputs only when they differ from the pinned settings.",
    )
    args = parser.parse_args()
    if args.check and args.force:
        parser.error("--check and --force are mutually exclusive")
    return args


def input_text(spec: AtomSpec) -> str:
    return (
        f"! {FUNCTIONAL_BASIS} EnGrad RIJCOSX\n"
        f"%pal nprocs {ORCA_NPROCS} end\n"
        f"* xyz {spec.charge} {spec.multiplicity}\n"
        f"{spec.symbol} 0.000000000000 0.000000000000 0.000000000000\n"
        "*\n"
    )


def manifest_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for spec in ATOM_SPECS:
        stem = f"orcaatomizationD4rev{spec.symbol}"
        rows.append(
            {
                "atom": spec.symbol,
                "charge": spec.charge,
                "multiplicity": spec.multiplicity,
                "functional": "wB97M-D4rev",
                "basis": "def2-TZVPD",
                "nprocs": ORCA_NPROCS,
                "run_dir": f"runs/{spec.symbol}",
                "input_path": f"runs/{spec.symbol}/{stem}.inp",
                "output_path": f"runs/{spec.symbol}/{stem}.out",
            }
        )
    return rows


def expected_manifest_text() -> str:
    fieldnames = list(manifest_rows()[0])
    lines = [",".join(fieldnames)]
    for row in manifest_rows():
        lines.append(",".join(str(row[field]) for field in fieldnames))
    return "\n".join(lines) + "\n"


def validate_manifest() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    expected = [{key: str(value) for key, value in row.items()} for row in manifest_rows()]
    if actual != expected:
        raise ValueError(f"Manifest does not match the pinned D4rev atom specifications: {MANIFEST_PATH}")


def check_inputs() -> None:
    validate_manifest()
    for spec in ATOM_SPECS:
        path = RUNS_DIR / spec.symbol / f"orcaatomizationD4rev{spec.symbol}.inp"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ORCA input: {path}")
        if path.read_text(encoding="utf-8") != input_text(spec):
            raise ValueError(
                f"ORCA input differs from the pinned functional, basis, charge, "
                f"multiplicity, or coordinates: {path}"
            )
        print(
            f"validated {path.relative_to(SCRIPT_DIR)} "
            f"(charge={spec.charge}, multiplicity={spec.multiplicity})"
        )


def write_inputs(force: bool) -> None:
    for spec in ATOM_SPECS:
        path = RUNS_DIR / spec.symbol / f"orcaatomizationD4rev{spec.symbol}.inp"
        expected = input_text(spec)
        if path.exists():
            actual = path.read_text(encoding="utf-8")
            if actual == expected:
                print(f"unchanged {path.relative_to(SCRIPT_DIR)}")
                continue
            if not force:
                raise FileExistsError(f"Input differs; inspect it or pass --force to replace: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8", newline="\n")
        print(
            f"wrote {path.relative_to(SCRIPT_DIR)} "
            f"(charge={spec.charge}, multiplicity={spec.multiplicity})"
        )

    expected_manifest = expected_manifest_text()
    if MANIFEST_PATH.exists() and MANIFEST_PATH.read_text(encoding="utf-8") != expected_manifest:
        if not force:
            raise FileExistsError(f"Manifest differs; inspect it or pass --force to replace: {MANIFEST_PATH}")
    MANIFEST_PATH.write_text(expected_manifest, encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_PATH.relative_to(SCRIPT_DIR)}")


def main() -> None:
    args = parse_args()
    if args.check:
        check_inputs()
    else:
        write_inputs(force=args.force)
        check_inputs()


if __name__ == "__main__":
    main()
