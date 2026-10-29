#!/usr/bin/env python3
"""Strict charge-safe validation of ORCA EnGrad outputs for this campaign."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms


FORMAL_CHARGES = {"H": 1, "N": -3, "O": -2, "S": -2}
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / 0.529177210903
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
XYZ_RE = re.compile(r"^\s*\*\s*xyz\s+([+-]?\d+)\s+(\d+)\s*$", re.I | re.M)
ROUTE_RE = re.compile(r"^\s*!\s+(.+?)\s+EnGrad\s+RIJCOSX\s*$", re.M)
ATOM_RE = re.compile(rf"^\s*([A-Za-z]+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
OUTPUT_CHARGE_RE = re.compile(r"Total\s+Charge\s+Charge\s+\.+\s+([+-]?\d+)", re.I)
OUTPUT_XYZ_RE = re.compile(r"^\s*\|\s*\d+>\s*\*\s*xyz\s+([+-]?\d+)\s+(\d+)\s*$", re.I | re.M)
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
    expected_functional_basis: str


def parse_input(path: Path) -> tuple[int, int, list[str], np.ndarray, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    route = ROUTE_RE.search(text)
    if route is None:
        raise ValueError("missing strict '! FUNCTIONAL BASIS EnGrad RIJCOSX' route")
    functional_basis = route.group(1).strip()
    match = XYZ_RE.search(text)
    if match is None:
        raise ValueError("missing * xyz charge multiplicity declaration")
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
        raise ValueError("empty * xyz block")
    return charge, spin, symbols, np.asarray(positions), functional_basis


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
    input_charge = input_spin = calculated_charge = None
    functional_basis = None
    symbols: list[str] = []
    positions = np.empty((0, 3))
    try:
        input_charge, input_spin, symbols, positions, functional_basis = parse_input(job.input_path)
        if functional_basis != job.expected_functional_basis:
            issues.append("input_functional_basis_mismatch")
        unknown = sorted(set(symbols) - FORMAL_CHARGES.keys())
        if unknown:
            issues.append("unsupported_elements:" + ",".join(unknown))
        else:
            calculated_charge = sum(FORMAL_CHARGES[symbol] for symbol in symbols)
        if input_charge != job.manifest_charge:
            issues.append("manifest_input_charge_mismatch")
        if calculated_charge is not None and input_charge != calculated_charge:
            issues.append("input_formal_charge_mismatch")
        if input_spin != job.manifest_spin:
            issues.append("manifest_input_spin_mismatch")
        if input_spin != 1 or job.manifest_spin != 1:
            issues.append("non_singlet_multiplicity")
        if calculated_charge is not None and (sum(Atoms(symbols).numbers) - calculated_charge) % 2:
            issues.append("odd_electron_count_for_required_singlet")
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
        echoed_inputs = OUTPUT_XYZ_RE.findall(text)
        if echoed_inputs:
            output_input_charge, output_input_spin = map(int, echoed_inputs[-1])
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
        elif calculated_charge is not None and output_charge != calculated_charge:
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
        atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
        atoms.info.update({
            "REF_energy": energy_hartree * HARTREE_TO_EV,
            "charge": calculated_charge,
            "spin": input_spin,
            "config_type": "ORCA_DFT",
            "source_file": job.output_path.name,
            "source_frame": job.frame,
            "formal_charge_verified": True,
            "dft_functional_basis": functional_basis,
        })
        atoms.arrays["REF_forces"] = -gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM

    row: dict[str, object] = {
        "frame": job.frame, "stem": job.stem,
        "status": "accepted" if accepted else "rejected", "issues": ";".join(issues),
        "manifest_charge": job.manifest_charge,
        "input_charge": "" if input_charge is None else input_charge,
        "formal_charge": "" if calculated_charge is None else calculated_charge,
        "output_charge": "" if output_charge is None else output_charge,
        "output_input_charge": "" if output_input_charge is None else output_input_charge,
        "output_input_spin": "" if output_input_spin is None else output_input_spin,
        "manifest_spin": job.manifest_spin,
        "input_spin": "" if input_spin is None else input_spin,
        "manifest_natoms": job.manifest_natoms, "parsed_natoms": len(symbols),
        "normal_termination": normal,
        "energy_hartree": "" if energy_hartree is None else energy_hartree,
        "expected_functional_basis": job.expected_functional_basis,
        "input_functional_basis": "" if functional_basis is None else functional_basis,
        "input_path": str(job.input_path), "output_path": str(job.output_path),
    }
    return row, atoms
