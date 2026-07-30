#!/usr/bin/env python3
"""Convert ORCA .out files into MACE/ASE extended XYZ training data."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "7_25_h2formationorca"
DEFAULT_OUTPUT = SCRIPT_DIR / "h2formation_orca.xyz"


# Typical ORCA output:
# Total Charge           Charge ....    0
# Multiplicity           Mult   ....    1
CHARGE_RE = re.compile(
    r"Total\s+Charge\s+Charge\s+\.+\s+([+-]?\d+)",
    flags=re.IGNORECASE,
)
MULTIPLICITY_RE = re.compile(
    r"Multiplicity\s+Mult\s+\.+\s+(\d+)",
    flags=re.IGNORECASE,
)

# Fallback: an echoed ORCA coordinate declaration such as "* xyz 0 1"
XYZ_DECLARATION_RE = re.compile(
    r"^\s*\*\s*xyz(?:file)?\s+([+-]?\d+)\s+(\d+)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def parse_charge_and_multiplicity(
    output_path: Path,
    charge_override: int | None,
    multiplicity_override: int | None,
) -> tuple[int, int]:
    """Extract total charge and multiplicity from an ORCA output."""
    text = output_path.read_text(encoding="utf-8", errors="replace")

    charge_matches = CHARGE_RE.findall(text)
    multiplicity_matches = MULTIPLICITY_RE.findall(text)

    charge = (
        charge_override
        if charge_override is not None
        else int(charge_matches[-1])
        if charge_matches
        else None
    )
    multiplicity = (
        multiplicity_override
        if multiplicity_override is not None
        else int(multiplicity_matches[-1])
        if multiplicity_matches
        else None
    )

    if charge is None or multiplicity is None:
        xyz_matches = XYZ_DECLARATION_RE.findall(text)
        if xyz_matches:
            xyz_charge, xyz_multiplicity = xyz_matches[-1]
            if charge is None:
                charge = int(xyz_charge)
            if multiplicity is None:
                multiplicity = int(xyz_multiplicity)

    if charge is None:
        raise ValueError(
            f"Could not determine total charge from {output_path}. "
            "Supply --charge explicitly."
        )
    if multiplicity is None:
        raise ValueError(
            f"Could not determine spin multiplicity from {output_path}. "
            "Supply --multiplicity explicitly."
        )

    return charge, multiplicity


def extract_labelled_frames(
    output_path: Path,
) -> tuple[list[tuple[int, Atoms, float, np.ndarray]], int]:
    """Return ORCA frames that contain both finite energy and forces."""
    images = read(output_path, index=":", format="orca-output")
    if isinstance(images, Atoms):
        images = [images]

    usable: list[tuple[int, Atoms, float, np.ndarray]] = []

    for step, atoms in enumerate(images):
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
        except (
            PropertyNotImplementedError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
        ):
            continue

        if forces.shape != (len(atoms), 3):
            print(
                f"Warning: ignoring {output_path}, step {step}: "
                f"force shape is {forces.shape}, expected {(len(atoms), 3)}.",
                file=sys.stderr,
            )
            continue

        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            print(
                f"Warning: ignoring {output_path}, step {step}: "
                "non-finite energy or forces.",
                file=sys.stderr,
            )
            continue

        usable.append((step, atoms, energy, forces))

    return usable, len(images)


def make_training_atoms(
    atoms: Atoms,
    *,
    energy: float,
    forces: np.ndarray,
    charge: int,
    multiplicity: int,
    config_type: str,
    source_file: str,
    orca_step: int,
) -> Atoms:
    """Create a calculator-free Atoms object with explicit training labels."""
    converted = atoms.copy()

    # Avoid ASE writing calculator results under conflicting key names.
    converted.calc = None

    # ORCA molecular calculations are nonperiodic.
    converted.set_pbc(False)

    # Graph-level labels.
    converted.info["REF_energy"] = float(energy)
    converted.info["charge"] = int(charge)
    converted.info["spin"] = int(multiplicity)
    converted.info["external_field"] = [0.0, 0.0, 0.0]
    converted.info["config_type"] = config_type

    # Provenance metadata; MACE ignores these unless explicitly configured.
    converted.info["source_file"] = source_file
    converted.info["orca_step"] = int(orca_step)

    # Atom-level labels.
    converted.arrays["REF_forces"] = np.asarray(forces, dtype=float).copy()

    return converted


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    """Expand input files/directories and deduplicate the resulting .out files."""
    requested = inputs or [DEFAULT_INPUT_DIR]
    resolved: set[Path] = set()

    for input_path in requested:
        if input_path.is_dir():
            resolved.update(path.resolve() for path in input_path.glob("*.out"))
            continue

        # Python receives wildcard arguments literally in shells such as
        # PowerShell, so expand them here as well.
        if any(character in str(input_path) for character in "*?["):
            parent = input_path.parent if str(input_path.parent) else Path(".")
            resolved.update(
                path.resolve()
                for path in parent.glob(input_path.name)
                if path.is_file()
            )
            continue

        resolved.add(input_path.resolve())

    return sorted(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ORCA outputs to MACE extended XYZ."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "ORCA .out files, directories, or wildcard patterns. Defaults to "
            f"{DEFAULT_INPUT_DIR}."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output extended-XYZ file. Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help=(
            "For geometry optimizations, retain every step having energy and "
            "forces. By default, retain only the last usable step per file."
        ),
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help="Override the charge parsed from every input file.",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help="Override the spin multiplicity parsed from every input file.",
    )
    parser.add_argument(
        "--config-type",
        default="ORCA_DFT",
        help="Value assigned to the extxyz config_type field.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow output files without 'ORCA TERMINATED NORMALLY'.",
    )

    args = parser.parse_args()

    converted_frames: list[Atoms] = []

    input_paths = resolve_inputs(args.inputs)
    if not input_paths:
        raise RuntimeError("No .out input files were found.")

    for output_path in input_paths:
        if not output_path.is_file():
            print(f"Warning: file not found: {output_path}", file=sys.stderr)
            continue

        text = output_path.read_text(encoding="utf-8", errors="replace")
        if (
            "ORCA TERMINATED NORMALLY" not in text
            and not args.allow_incomplete
        ):
            print(
                f"Skipping {output_path}: ORCA did not terminate normally.",
                file=sys.stderr,
            )
            continue

        try:
            charge, multiplicity = parse_charge_and_multiplicity(
                output_path,
                charge_override=args.charge,
                multiplicity_override=args.multiplicity,
            )

            usable, total_frames = extract_labelled_frames(output_path)
        except Exception as exc:
            print(f"Skipping {output_path}: {exc}", file=sys.stderr)
            continue

        if not usable:
            print(
                f"Skipping {output_path}: no frame contained both an energy "
                "and Cartesian gradient. Run a single-point ENGRAD calculation "
                "on the desired geometry.",
                file=sys.stderr,
            )
            continue

        selected = usable if args.all_steps else [usable[-1]]

        for step, atoms, energy, forces in selected:
            converted_frames.append(
                make_training_atoms(
                    atoms,
                    energy=energy,
                    forces=forces,
                    charge=charge,
                    multiplicity=multiplicity,
                    config_type=args.config_type,
                    source_file=output_path.name,
                    orca_step=step,
                )
            )

        print(
            f"{output_path}: read {total_frames} frame(s), "
            f"found {len(usable)} with energy+forces, "
            f"wrote {len(selected)}.",
            file=sys.stderr,
        )

    if not converted_frames:
        raise RuntimeError("No valid configurations were found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, converted_frames, format="extxyz")

    print(
        f"Wrote {len(converted_frames)} configurations to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
