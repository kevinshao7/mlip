#!/usr/bin/env python3
"""Validate and condense C3 ORCA outputs into one charge-safe extxyz dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DFT_DIR = MLIP_DIR / "outputsfull" / "C3_DFTproductionstopH2_ON" / "dft_outputs"
DEFAULT_MANIFEST = SCRIPT_DIR / "expand" / "manifest.csv"
DEFAULT_OUTPUT_DIR = MLIP_DIR / "outputsfull" / "C3_DFTproductionstopH2_ON" / "processed_dft_outputs"
FORMAL_CHARGES = {"H": 1, "N": -3, "O": -2, "S": -2}
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / 0.529177210903
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
XYZ_RE = re.compile(r"^\s*\*xyz\s+([+-]?\d+)\s+(\d+)\s*$", re.I | re.M)
ATOM_RE = re.compile(rf"^\s*([A-Za-z]+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
OUTPUT_CHARGE_RE = re.compile(r"Total\s+Charge\s+Charge\s+\.+\s+([+-]?\d+)", re.I)
# ORCA echoes the input deck near the beginning of every output.  This is the
# authoritative record of what was actually calculated; the current .inp may
# have been regenerated since the output was produced.
OUTPUT_XYZ_RE = re.compile(r"^\s*\|\s*\d+>\s*\*xyz\s+([+-]?\d+)\s+(\d+)\s*$", re.I | re.M)
ENERGY_RE = re.compile(rf"FINAL SINGLE POINT ENERGY\s+({FLOAT})")
GRADIENT_RE = re.compile(rf"^\s*\d+\s+([A-Za-z]+)\s*:\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")


@dataclass(frozen=True)
class Job:
    frame: int
    stem: str
    input_path: Path
    output_path: Path
    manifest_charge: int
    manifest_spin: int
    manifest_natoms: int


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dft-dir", type=Path, default=DEFAULT_DFT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless every manifest frame is accepted.")
    return parser.parse_args()


def read_manifest(path: Path, dft_dir: Path) -> list[Job]:
    jobs: list[Job] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            input_path = SCRIPT_DIR / row["input"]
            jobs.append(Job(
                frame=int(row["frame"]), stem=row["stem"], input_path=input_path,
                output_path=dft_dir / f"{row['stem']}.out", manifest_charge=int(row["charge"]),
                manifest_spin=int(row["multiplicity"]), manifest_natoms=int(row["n_atoms"]),
            ))
    if not jobs:
        raise ValueError(f"No jobs found in {path}")
    return jobs


def parse_input(path: Path) -> tuple[int, int, list[str], np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = XYZ_RE.search(text)
    if match is None:
        raise ValueError("missing *xyz declaration")
    charge, spin = map(int, match.groups())
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in text[match.end():].splitlines():
        if not line.strip():
            continue
        if line.strip() == "*":
            break
        atom = ATOM_RE.match(line)
        if atom is None:
            raise ValueError(f"malformed coordinate line: {line!r}")
        symbol, x, y, z = atom.groups()
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    if not symbols:
        raise ValueError("empty *xyz block")
    return charge, spin, symbols, np.asarray(positions)


def last_gradient(text: str) -> tuple[list[str], np.ndarray] | None:
    blocks: list[list[tuple[str, list[float]]]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "CARTESIAN GRADIENT":
            continue
        block: list[tuple[str, list[float]]] = []
        for candidate in lines[index + 1:]:
            match = GRADIENT_RE.match(candidate)
            if match:
                symbol, x, y, z = match.groups()
                block.append((symbol, [float(x), float(y), float(z)]))
            elif block:
                break
        if block:
            blocks.append(block)
    if not blocks:
        return None
    return [row[0] for row in blocks[-1]], np.asarray([row[1] for row in blocks[-1]])


def process(job: Job) -> tuple[dict[str, object], Atoms | None]:
    issues: list[str] = []
    input_charge = input_spin = formal_charge = None
    symbols: list[str] = []
    positions = np.empty((0, 3))
    try:
        input_charge, input_spin, symbols, positions = parse_input(job.input_path)
        unknown = sorted(set(symbols) - FORMAL_CHARGES.keys())
        if unknown:
            issues.append("unsupported_elements:" + ",".join(unknown))
        else:
            formal_charge = sum(FORMAL_CHARGES[symbol] for symbol in symbols)
        if input_charge != job.manifest_charge:
            issues.append("manifest_input_charge_mismatch")
        if formal_charge is not None and input_charge != formal_charge:
            issues.append("input_formal_charge_mismatch")
        if input_spin != job.manifest_spin:
            issues.append("manifest_input_spin_mismatch")
        if len(symbols) != job.manifest_natoms:
            issues.append("manifest_input_natoms_mismatch")
    except Exception as exc:
        issues.append(f"input_error:{exc}")

    output_charge = output_input_charge = output_input_spin = energy_hartree = None
    normal = False
    gradient = None
    if not job.output_path.is_file():
        issues.append("missing_output")
    else:
        text = job.output_path.read_text(encoding="utf-8", errors="replace")
        output_inputs = OUTPUT_XYZ_RE.findall(text)
        if output_inputs:
            output_input_charge, output_input_spin = map(int, output_inputs[-1])
            if input_charge is not None and output_input_charge != input_charge:
                issues.append("stale_output_input_charge")
            if input_spin is not None and output_input_spin != input_spin:
                issues.append("stale_output_input_spin")
        else:
            issues.append("missing_output_input_provenance")
        normal = "ORCA TERMINATED NORMALLY" in text
        if not normal:
            issues.append("no_normal_termination")
        charges = OUTPUT_CHARGE_RE.findall(text)
        output_charge = int(charges[-1]) if charges else None
        if output_charge is None:
            issues.append("missing_output_charge")
        elif input_charge is not None and output_charge != input_charge:
            issues.append("output_input_charge_mismatch")
        elif formal_charge is not None and output_charge != formal_charge:
            issues.append("output_formal_charge_mismatch")
        energies = ENERGY_RE.findall(text)
        energy_hartree = float(energies[-1]) if energies else None
        if energy_hartree is None:
            issues.append("missing_energy")
        parsed_gradient = last_gradient(text)
        if parsed_gradient is None:
            issues.append("missing_gradient")
        else:
            gradient_symbols, gradient = parsed_gradient
            if gradient_symbols != symbols:
                issues.append("gradient_symbol_or_count_mismatch")

    accepted = not issues
    atoms = None
    if accepted:
        forces = -gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
        atoms.info.update({
            "REF_energy": energy_hartree * HARTREE_TO_EV, "charge": formal_charge,
            "spin": input_spin, "config_type": "ORCA_DFT", "source_file": job.output_path.name,
            "source_frame": job.frame, "formal_charge_verified": True,
        })
        atoms.arrays["REF_forces"] = forces

    row: dict[str, object] = {
        "frame": job.frame, "stem": job.stem, "status": "accepted" if accepted else "rejected",
        "issues": ";".join(issues), "manifest_charge": job.manifest_charge,
        "input_charge": "" if input_charge is None else input_charge,
        "formal_charge": "" if formal_charge is None else formal_charge,
        "output_charge": "" if output_charge is None else output_charge,
        "output_input_charge": "" if output_input_charge is None else output_input_charge,
        "output_input_spin": "" if output_input_spin is None else output_input_spin,
        "manifest_spin": job.manifest_spin, "input_spin": "" if input_spin is None else input_spin,
        "manifest_natoms": job.manifest_natoms, "parsed_natoms": len(symbols),
        "normal_termination": normal, "energy_hartree": "" if energy_hartree is None else energy_hartree,
        "input_path": str(job.input_path), "output_path": str(job.output_path),
    }
    return row, atoms


def main() -> None:
    args = arguments()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    jobs = read_manifest(args.manifest.resolve(), args.dft_dir.resolve())
    if args.workers == 1:
        results = [process(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            results = list(executor.map(process, jobs))
    rows = [result[0] for result in results]
    frames = [result[1] for result in results if result[1] is not None]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "charge_and_completion_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if frames:
        write(args.output_dir / "target_all.extxyz", frames, format="extxyz")
    rejected = [row for row in rows if row["status"] == "rejected"]
    issue_counts = Counter(issue for row in rejected for issue in str(row["issues"]).split(";") if issue)
    stats = {
        "manifest_frames": len(jobs), "accepted_frames": len(frames), "rejected_frames": len(rejected),
        "formal_charge_rules": FORMAL_CHARGES, "issue_counts": dict(issue_counts),
        "output_extxyz": str(args.output_dir / "target_all.extxyz"), "audit_csv": str(audit_path),
    }
    with (args.output_dir / "condensation_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))
    if args.strict and rejected:
        raise SystemExit(f"Strict validation failed: {len(rejected)} of {len(jobs)} frames rejected")


if __name__ == "__main__":
    main()
