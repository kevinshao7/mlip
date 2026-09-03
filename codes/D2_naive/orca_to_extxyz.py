#!/usr/bin/env python3
"""Convert ORCA output files into MACE/ASE extended XYZ training data."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_DIR = (
    MLIP_DIR / "outputsfull" / "C_DFTproduction" / "C_DFTproduction" / "dft_outputs"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "target_all.xyz"
DEFAULT_WORKERS = 8

CHARGE_RE = re.compile(
    r"Total\s+Charge\s+Charge\s+\.+\s+([+-]?\d+)", flags=re.IGNORECASE
)
MULTIPLICITY_RE = re.compile(
    r"Multiplicity\s+Mult\s+\.+\s+(\d+)", flags=re.IGNORECASE
)
XYZ_DECLARATION_RE = re.compile(
    r"^\s*\*\s*xyz(?:file)?\s+([+-]?\d+)\s+(\d+)",
    flags=re.IGNORECASE | re.MULTILINE,
)


def parse_charge_and_multiplicity(
    text: str,
    output_path: Path,
    charge_override: int | None,
    multiplicity_override: int | None,
) -> tuple[int, int]:
    charge_matches = CHARGE_RE.findall(text)
    multiplicity_matches = MULTIPLICITY_RE.findall(text)

    charge = charge_override
    if charge is None and charge_matches:
        charge = int(charge_matches[-1])

    multiplicity = multiplicity_override
    if multiplicity is None and multiplicity_matches:
        multiplicity = int(multiplicity_matches[-1])

    if charge is None or multiplicity is None:
        xyz_matches = XYZ_DECLARATION_RE.findall(text)
        if xyz_matches:
            xyz_charge, xyz_multiplicity = xyz_matches[-1]
            if charge is None:
                charge = int(xyz_charge)
            if multiplicity is None:
                multiplicity = int(xyz_multiplicity)

    if charge is None:
        raise ValueError(f"Could not determine charge from {output_path}")
    if multiplicity is None:
        raise ValueError(f"Could not determine spin multiplicity from {output_path}")
    return charge, multiplicity


def extract_labelled_frames(output_path: Path) -> tuple[list[tuple[int, Atoms, float, np.ndarray]], int]:
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
                f"Warning: ignoring {output_path}, step {step}: force shape "
                f"{forces.shape}, expected {(len(atoms), 3)}",
                file=sys.stderr,
            )
            continue
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            print(
                f"Warning: ignoring {output_path}, step {step}: non-finite labels",
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
    vacuum: float,
) -> Atoms:
    converted = atoms.copy()
    converted.calc = None
    converted.set_pbc(False)

    if vacuum > 0:
        converted.center(vacuum=vacuum)

    converted.info["REF_energy"] = float(energy)
    converted.info["charge"] = int(charge)
    converted.info["spin"] = int(multiplicity)
    converted.info["external_field"] = [0.0, 0.0, 0.0]
    converted.info["config_type"] = config_type
    converted.info["source_file"] = source_file
    converted.info["orca_step"] = int(orca_step)
    converted.arrays["REF_forces"] = np.asarray(forces, dtype=float).copy()
    return converted


def resolve_inputs(inputs: list[Path]) -> list[Path]:
    requested = inputs or [DEFAULT_INPUT_DIR]
    resolved: set[Path] = set()
    for input_path in requested:
        if input_path.is_dir():
            resolved.update(path.resolve() for path in input_path.glob("*.out"))
            continue
        if any(character in str(input_path) for character in "*?["):
            parent = input_path.parent if str(input_path.parent) else Path(".")
            resolved.update(
                path.resolve() for path in parent.glob(input_path.name) if path.is_file()
            )
            continue
        resolved.add(input_path.resolve())
    return sorted(resolved)


def convert_one_output(
    output_path: Path,
    *,
    charge: int | None,
    multiplicity: int | None,
    config_type: str,
    all_steps: bool,
    allow_incomplete: bool,
    vacuum: float,
) -> tuple[Path, list[Atoms], str | None]:
    try:
        if not output_path.is_file():
            return output_path, [], f"file not found: {output_path}"
        text = output_path.read_text(encoding="utf-8", errors="replace")
        if "ORCA TERMINATED NORMALLY" not in text and not allow_incomplete:
            return output_path, [], "ORCA did not terminate normally"

        parsed_charge, parsed_multiplicity = parse_charge_and_multiplicity(
            text, output_path, charge, multiplicity
        )
        usable, total_frames = extract_labelled_frames(output_path)
        if not usable:
            return output_path, [], "no frame contained both energy and forces"

        selected = usable if all_steps else [usable[-1]]
        converted_frames: list[Atoms] = []
        for step, atoms, energy, forces in selected:
            converted_frames.append(
                make_training_atoms(
                    atoms,
                    energy=energy,
                    forces=forces,
                    charge=parsed_charge,
                    multiplicity=parsed_multiplicity,
                    config_type=config_type,
                    source_file=output_path.name,
                    orca_step=step,
                    vacuum=vacuum,
                )
            )
        message = f"read {total_frames}, labelled {len(usable)}, wrote {len(selected)}"
        return output_path, converted_frames, message
    except Exception as exc:  # pylint: disable=broad-except
        return output_path, [], str(exc)


def _convert_one_output_star(payload: tuple[Path, dict]) -> tuple[Path, list[Atoms], str | None]:
    output_path, kwargs = payload
    return convert_one_output(output_path, **kwargs)


def convert_outputs(args: argparse.Namespace) -> int:
    converted_frames: list[Atoms] = []
    skipped = 0

    input_paths = resolve_inputs(args.inputs)
    if not input_paths:
        raise RuntimeError("No .out input files were found.")

    kwargs = {
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "config_type": args.config_type,
        "all_steps": args.all_steps,
        "allow_incomplete": args.allow_incomplete,
        "vacuum": args.vacuum,
    }
    workers = max(1, int(args.workers))
    print(
        f"Converting {len(input_paths)} ORCA output file(s) with {workers} worker(s)",
        file=sys.stderr,
    )

    if workers == 1 or len(input_paths) == 1:
        results = [convert_one_output(path, **kwargs) for path in input_paths]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_convert_one_output_star, [(path, kwargs) for path in input_paths])

    failures: list[str] = []
    for output_path, frames, message in results:
        if frames:
            converted_frames.extend(frames)
            print(f"{output_path}: {message}", file=sys.stderr)
        else:
            skipped += 1
            failure = f"{output_path}: {message}"
            failures.append(failure)
            print(f"FAILED {failure}", file=sys.stderr)

    if not converted_frames:
        raise RuntimeError("No valid configurations were found.")
    if failures and args.strict:
        formatted = "\n  ".join(failures)
        raise RuntimeError(
            "One or more ORCA outputs failed conversion. Refusing to write a "
            f"partial dataset because --strict was passed:\n  {formatted}"
        )
    if failures:
        formatted = "\n  ".join(failures)
        print(
            "WARNING: writing partial dataset; these ORCA outputs were not "
            f"written:\n  {formatted}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, converted_frames, format="extxyz")
    print(
        f"Wrote {len(converted_frames)} configurations to {args.output}; skipped {skipped}",
        file=sys.stderr,
    )
    return len(converted_frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="ORCA .out files, dirs, or globs.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--multiplicity", type=int, default=None)
    parser.add_argument("--config-type", default="ORCA_DFT")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of writing a partial dataset if any ORCA output fails.",
    )
    parser.add_argument("--vacuum", type=float, default=0.0, help="Optional recentering vacuum in Angstrom.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel ORCA parser processes.")
    return parser


def main() -> None:
    convert_outputs(build_parser().parse_args())


if __name__ == "__main__":
    main()

