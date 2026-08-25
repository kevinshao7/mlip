#!/usr/bin/env python3
"""Process ORCA DFT single-point outputs into parity-plot reference files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
DEFAULT_LATTICE = (24.0, 24.0, 24.0)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DFT_DIR = (
    REPO_ROOT
    / "outputsfull"
    / "C_DFTproduction"
    / "C_DFTproduction"
    / "dft_outputs"
)
DEFAULT_GENERATION_DIR = REPO_ROOT / "codes" / "C_DFTproduction"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputsfull"
    / "C_DFTproduction"
    / "C_DFTproduction"
    / "processed_dft_outputs"
)

FRAME_RE = re.compile(r"_(\d+)\.(?:out|inp)$")
FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
XYZ_BLOCK_HEADER_RE = re.compile(r"^\*xyz\s+(-?\d+)\s+(\d+)\s*$", re.IGNORECASE)
XYZ_BLOCK_ATOM_RE = re.compile(
    r"^\s*([A-Za-z]+)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
GRADIENT_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+:\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
ORCA_NORMAL_TERMINATION = "ORCA TERMINATED NORMALLY"


@dataclass
class FrameMetadata:
    frame: int
    stem: str
    input_path: Path | None
    output_path: Path | None
    charge: int | None
    multiplicity: int | None
    n_atoms_manifest: int | None


@dataclass
class StructureData:
    symbols: list[str]
    positions: list[list[float]]
    charge: int | None
    multiplicity: int | None

    def __len__(self) -> int:
        return len(self.symbols)


def status(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dft-dir", type=Path, default=DEFAULT_DFT_DIR)
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV. Defaults to <generation-dir>/expand/manifest.csv.",
    )
    parser.add_argument(
        "--lattice",
        default="24,24,24",
        help="Cell lengths in Angstrom for the extxyz output. Default: 24,24,24.",
    )
    parser.add_argument(
        "--prefix",
        default="C_DFTproduction",
        help="Prefix for generated files. Default: C_DFTproduction.",
    )
    parser.add_argument("--allow-missing-input", action="store_true", help="Keep frame summaries even if .inp is missing.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_lattice(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated lattice lengths")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def frame_from_name(name: str) -> int:
    match = FRAME_RE.search(name)
    if not match:
        raise ValueError(f"Could not parse frame index from {name}")
    return int(match.group(1))


def discover_metadata(dft_dir: Path, generation_dir: Path, manifest_path: Path | None) -> dict[int, FrameMetadata]:
    manifest = manifest_path if manifest_path is not None else generation_dir / "expand" / "manifest.csv"
    frames: dict[int, FrameMetadata] = {}

    if manifest.is_file():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame = int(row["frame"])
                stem = row["stem"]
                input_rel = row.get("input") or ""
                frames[frame] = FrameMetadata(
                    frame=frame,
                    stem=stem,
                    input_path=(generation_dir / input_rel) if input_rel else None,
                    output_path=dft_dir / f"{stem}.out",
                    charge=int(row["charge"]) if row.get("charge") else None,
                    multiplicity=int(row["multiplicity"]) if row.get("multiplicity") else None,
                    n_atoms_manifest=int(row["n_atoms"]) if row.get("n_atoms") else None,
                )

    for out_path in sorted(dft_dir.glob("*.out")):
        frame = frame_from_name(out_path.name)
        entry = frames.get(frame)
        if entry is None:
            stem = out_path.stem
            guessed_input = generation_dir / "expand" / f"{stem}.inp"
            frames[frame] = FrameMetadata(
                frame=frame,
                stem=stem,
                input_path=guessed_input if guessed_input.is_file() else None,
                output_path=out_path,
                charge=None,
                multiplicity=None,
                n_atoms_manifest=None,
            )
        else:
            entry.output_path = out_path
    return dict(sorted(frames.items()))


def atoms_from_orca_input(path: Path) -> StructureData:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    xyz_start = None
    charge = None
    multiplicity = None
    for line_index, line in enumerate(lines):
        match = XYZ_BLOCK_HEADER_RE.match(line.strip())
        if match:
            charge = int(match.group(1))
            multiplicity = int(match.group(2))
            xyz_start = line_index + 1
            break
    if xyz_start is None:
        raise ValueError(f"No *xyz block found in {path}")

    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[xyz_start:]:
        stripped = line.strip()
        if stripped == "*":
            break
        match = XYZ_BLOCK_ATOM_RE.match(line)
        if not match:
            raise ValueError(f"Malformed XYZ line in {path}: {line}")
        symbol, x, y, z = match.groups()
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    if not symbols:
        raise ValueError(f"Empty *xyz block in {path}")
    return StructureData(symbols=symbols, positions=positions, charge=charge, multiplicity=multiplicity)


def parse_energy_hartree(text: str) -> float | None:
    matches = FINAL_ENERGY_RE.findall(text)
    return float(matches[-1]) if matches else None


def parse_gradient_block(text: str) -> tuple[list[str], list[list[float]]] | None:
    lines = text.splitlines()
    blocks: list[list[tuple[str, list[float]]]] = []
    for line_index, line in enumerate(lines):
        if line.strip() != "CARTESIAN GRADIENT":
            continue
        block: list[tuple[str, list[float]]] = []
        for gradient_line in lines[line_index + 1 :]:
            match = GRADIENT_LINE_RE.match(gradient_line)
            if match:
                _, symbol, gx, gy, gz = match.groups()
                block.append((symbol, [float(gx), float(gy), float(gz)]))
            elif block:
                break
        if block:
            blocks.append(block)
    if not blocks:
        return None
    last_block = blocks[-1]
    symbols = [symbol for symbol, _ in last_block]
    gradient = [values for _, values in last_block]
    return symbols, gradient


def safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def scale_vectors(vectors: list[list[float]], factor: float) -> list[list[float]]:
    return [[factor * component for component in row] for row in vectors]


def write_extxyz(
    path: Path,
    records: list[dict[str, Any]],
    lattice: tuple[float, float, float],
) -> None:
    lattice_text = f'{lattice[0]} 0.0 0.0 0.0 {lattice[1]} 0.0 0.0 0.0 {lattice[2]}'
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            symbols = record["symbols"]
            positions = record["positions"]
            forces = record["forces"]
            natoms = len(symbols)
            handle.write(f"{natoms}\n")
            info_parts = [
                f'Lattice="{lattice_text}"',
                "Properties=species:S:1:pos:R:3:forces:R:3",
                'pbc="F F F"',
                f'frame={record["frame"]}',
                f'source_stem="{record["stem"]}"',
                f'source_out="{record["source_out"]}"',
                f'energy={record["energy_eV"]:.12f}',
                f'energy_eV={record["energy_eV"]:.12f}',
                f'energy_hartree={record["energy_hartree"]:.12f}',
                f'orca_normal_termination={str(record["normal_termination"]).upper()}',
            ]
            if record["charge"] is not None:
                info_parts.append(f'charge={record["charge"]}')
            if record["multiplicity"] is not None:
                info_parts.append(f'spin={record["multiplicity"]}')
            handle.write(" ".join(info_parts) + "\n")
            for symbol, position, force in zip(symbols, positions, forces, strict=True):
                handle.write(
                    f"{symbol} "
                    f"{position[0]:.10f} {position[1]:.10f} {position[2]:.10f} "
                    f"{force[0]:.12f} {force[1]:.12f} {force[2]:.12f}\n"
                )


def main() -> None:
    args = parse_args()
    lattice = parse_lattice(args.lattice)

    if not args.dft_dir.is_dir():
        raise FileNotFoundError(f"DFT directory not found: {args.dft_dir}")
    if not args.generation_dir.is_dir():
        raise FileNotFoundError(f"Generation directory not found: {args.generation_dir}")

    metadata_by_frame = discover_metadata(args.dft_dir, args.generation_dir, args.manifest)
    if not metadata_by_frame:
        raise RuntimeError(f"No ORCA outputs or manifest entries found under {args.dft_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.prefix}_singlepoints.csv"
    forces_path = args.output_dir / f"{args.prefix}_forces.csv"
    extxyz_path = args.output_dir / f"{args.prefix}_complete.extxyz"
    stats_path = args.output_dir / f"{args.prefix}_stats.json"

    complete_extxyz_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    counters = {
        "total_frames": 0,
        "ok_frames": 0,
        "frames_with_output": 0,
        "frames_missing_output": 0,
        "frames_missing_input": 0,
        "frames_with_energy": 0,
        "frames_with_gradient": 0,
        "frames_normal_termination": 0,
        "frames_failed": 0,
    }

    for frame, meta in metadata_by_frame.items():
        counters["total_frames"] += 1
        issues: list[str] = []
        structure = None
        input_charge = meta.charge
        input_multiplicity = meta.multiplicity
        natoms_input = None

        if meta.input_path is None or not meta.input_path.is_file():
            counters["frames_missing_input"] += 1
            issues.append("missing_input")
            if not args.allow_missing_input:
                structure = None
            else:
                structure = None
        else:
            try:
                structure = atoms_from_orca_input(meta.input_path)
                natoms_input = len(structure)
                parsed_charge = structure.charge
                parsed_multiplicity = structure.multiplicity
                if input_charge is None:
                    input_charge = parsed_charge
                elif parsed_charge is not None and parsed_charge != input_charge:
                    issues.append("charge_mismatch_manifest_vs_input")
                if input_multiplicity is None:
                    input_multiplicity = parsed_multiplicity
                elif parsed_multiplicity is not None and parsed_multiplicity != input_multiplicity:
                    issues.append("multiplicity_mismatch_manifest_vs_input")
            except Exception as exc:
                issues.append(f"input_parse_error:{exc}")
                structure = None

        output_exists = meta.output_path is not None and meta.output_path.is_file()
        if output_exists:
            counters["frames_with_output"] += 1
            text = meta.output_path.read_text(encoding="utf-8", errors="ignore")
        else:
            counters["frames_missing_output"] += 1
            text = ""
            issues.append("missing_output")

        normal_termination = ORCA_NORMAL_TERMINATION in text
        if normal_termination:
            counters["frames_normal_termination"] += 1
        elif output_exists:
            issues.append("no_normal_termination")

        energy_hartree = parse_energy_hartree(text) if output_exists else None
        if energy_hartree is not None:
            counters["frames_with_energy"] += 1
        elif output_exists:
            issues.append("missing_energy")

        gradient_symbols = None
        gradient = None
        gradient_result = parse_gradient_block(text) if output_exists else None
        if gradient_result is not None:
            gradient_symbols, gradient = gradient_result
            counters["frames_with_gradient"] += 1
        elif output_exists:
            issues.append("missing_gradient")

        if meta.n_atoms_manifest is not None and natoms_input is not None and meta.n_atoms_manifest != natoms_input:
            issues.append("natoms_mismatch_manifest_vs_input")
        if structure is not None and gradient is not None:
            if len(structure) != len(gradient):
                issues.append("natoms_mismatch_input_vs_gradient")
            elif structure.symbols != gradient_symbols:
                issues.append("symbol_order_mismatch_input_vs_gradient")

        is_complete = structure is not None and energy_hartree is not None and gradient is not None
        status_value = "ok" if is_complete else "incomplete"
        if not is_complete:
            counters["frames_failed"] += 1
        else:
            counters["ok_frames"] += 1
            forces_ev_a = scale_vectors(gradient, -HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM)
            complete_extxyz_records.append(
                {
                    "frame": frame,
                    "stem": meta.stem,
                    "source_out": str(meta.output_path) if meta.output_path is not None else "",
                    "charge": input_charge,
                    "multiplicity": input_multiplicity,
                    "energy_eV": energy_hartree * HARTREE_TO_EV,
                    "energy_hartree": energy_hartree,
                    "normal_termination": normal_termination,
                    "symbols": structure.symbols,
                    "positions": structure.positions,
                    "forces": forces_ev_a,
                }
            )

            positions = structure.positions
            symbols = structure.symbols
            for atom_index, (symbol, position, grad_row, force_row) in enumerate(
                zip(symbols, positions, gradient, forces_ev_a, strict=True)
            ):
                force_rows.append(
                    {
                        "frame": frame,
                        "stem": meta.stem,
                        "atom_index": atom_index,
                        "symbol": symbol,
                        "x_A": f"{position[0]:.10f}",
                        "y_A": f"{position[1]:.10f}",
                        "z_A": f"{position[2]:.10f}",
                        "gradient_x_hartree_bohr": f"{grad_row[0]:.12f}",
                        "gradient_y_hartree_bohr": f"{grad_row[1]:.12f}",
                        "gradient_z_hartree_bohr": f"{grad_row[2]:.12f}",
                        "force_x_eV_A": f"{force_row[0]:.12f}",
                        "force_y_eV_A": f"{force_row[1]:.12f}",
                        "force_z_eV_A": f"{force_row[2]:.12f}",
                    }
                )

        summary_rows.append(
            {
                "frame": frame,
                "stem": meta.stem,
                "status": status_value,
                "issue_count": len(issues),
                "issues": safe_json(issues),
                "output_exists": int(output_exists),
                "normal_termination": int(normal_termination),
                "has_energy": int(energy_hartree is not None),
                "has_gradient": int(gradient is not None),
                "input_exists": int(meta.input_path is not None and meta.input_path.is_file()),
                "charge_e": "" if input_charge is None else input_charge,
                "multiplicity": "" if input_multiplicity is None else input_multiplicity,
                "n_atoms_manifest": "" if meta.n_atoms_manifest is None else meta.n_atoms_manifest,
                "n_atoms_input": "" if natoms_input is None else natoms_input,
                "n_atoms_gradient": "" if gradient is None else len(gradient),
                "energy_hartree": "" if energy_hartree is None else f"{energy_hartree:.12f}",
                "energy_eV": "" if energy_hartree is None else f"{energy_hartree * HARTREE_TO_EV:.12f}",
                "input_path": "" if meta.input_path is None else str(meta.input_path),
                "output_path": "" if meta.output_path is None else str(meta.output_path),
            }
        )

    if args.dry_run:
        status(f"Dry run: {counters['total_frames']} frames discovered, {counters['ok_frames']} complete.")
        return

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with forces_path.open("w", encoding="utf-8", newline="") as handle:
        if force_rows:
            fieldnames = list(force_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(force_rows)
        else:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame",
                    "stem",
                    "atom_index",
                    "symbol",
                    "x_A",
                    "y_A",
                    "z_A",
                    "gradient_x_hartree_bohr",
                    "gradient_y_hartree_bohr",
                    "gradient_z_hartree_bohr",
                    "force_x_eV_A",
                    "force_y_eV_A",
                    "force_z_eV_A",
                ]
            )

    if complete_extxyz_records:
        write_extxyz(extxyz_path, complete_extxyz_records, lattice)

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **counters,
                "dft_dir": str(args.dft_dir),
                "generation_dir": str(args.generation_dir),
                "output_dir": str(args.output_dir),
                "manifest": str(args.manifest) if args.manifest is not None else str(args.generation_dir / "expand" / "manifest.csv"),
                "lattice_A": lattice,
                "summary_csv": str(summary_path),
                "forces_csv": str(forces_path),
                "extxyz": str(extxyz_path) if complete_extxyz_records else "",
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    status(f"Wrote summary: {summary_path}")
    status(f"Wrote forces: {forces_path}")
    if complete_extxyz_records:
        status(f"Wrote extxyz: {extxyz_path}")
    status(f"Wrote stats: {stats_path}")
    status(
        f"Processed {counters['total_frames']} frames: "
        f"{counters['ok_frames']} complete, {counters['frames_failed']} incomplete."
    )


if __name__ == "__main__":
    main()
