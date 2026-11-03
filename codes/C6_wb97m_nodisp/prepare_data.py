"""Extract dispersion-free wB97M energy/forces from audited ORCA EnGrad stdout."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]
HARTREE = 27.211386245988
BOHR = 0.529177210903
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
ROW = re.compile(rf"^\s*(\d+)\s+([A-Z][a-z]?)\s*:\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
FORMAL = {"H": 1, "N": -3, "O": -2}


def one(pattern, text, label, flags=0):
    matches = re.findall(pattern, text, flags)
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one occurrence, got {len(matches)}")
    return matches[0]


def gradient(text, heading, symbols):
    tail = one(rf"^{heading}\s*\n(.*?)(?=Difference to translation invariance:)",
               text, heading, re.M | re.S)
    rows = [m.groups() for line in tail.splitlines() if (m := ROW.fullmatch(line))]
    if [int(r[0]) for r in rows] != list(range(1, len(symbols) + 1)):
        raise ValueError(f"{heading}: atom indices/count mismatch")
    if [r[1] for r in rows] != symbols:
        raise ValueError(f"{heading}: atom order mismatch")
    return np.asarray([[float(x) for x in r[2:]] for r in rows])


def convert(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if text.count("ORCA TERMINATED NORMALLY") != 1:
        raise ValueError("not exactly one normal termination")
    # The echoed input is retained inside stdout even when original .inp is unavailable.
    echoed = "\n".join(re.findall(r"^\s*\|\s*\d+> ?(.*)$", text, re.M))
    route = one(r"^!\s*(.+)$", echoed, "route", re.M).strip()
    if route.lower().split() != ["wb97m-d4rev", "def2-tzvpd", "engrad", "rijcosx"]:
        raise ValueError(f"unsupported method/options: {route}")
    charge, spin, coords = one(r"^\*\s*xyz\s+([+-]?\d+)\s+(\d+)\s*\n(.*?)^\*\s*$",
                              echoed, "inline input", re.M | re.S)
    symbols, positions = [], []
    for line in coords.splitlines():
        fields = line.split()
        if len(fields) != 4 or fields[0] not in FORMAL:
            raise ValueError(f"unsupported coordinate/element: {line}")
        symbols.append(fields[0])
        positions.append([float(x) for x in fields[1:]])
    if not symbols:
        raise ValueError("empty structure")
    formal = sum(FORMAL[s] for s in symbols)
    output_charge = int(one(r"Total\s+Charge\s+Charge\s+\.+\s+([+-]?\d+)", text, "output charge"))
    output_spin = int(one(r"Multiplicity\s+Mult\s+\.+\s+(\d+)", text, "output spin"))
    if int(charge) != formal or output_charge != formal:
        raise ValueError(f"formal/input/output charge mismatch: {formal}/{charge}/{output_charge}")
    atoms = Atoms(symbols, positions=positions, pbc=False)
    if int(spin) != output_spin or output_spin < 1:
        raise ValueError("input/output multiplicity mismatch")
    electrons = int(sum(atoms.numbers)) - formal
    if (electrons - (output_spin - 1)) % 2 or output_spin - 1 > electrons:
        raise ValueError("electron-count/multiplicity mismatch")
    total = float(one(rf"FINAL SINGLE POINT ENERGY\s+({FLOAT})", text, "energy")) * HARTREE
    dispersion = float(one(rf"^Dispersion correction\s+({FLOAT})\s*$", text, "D4 energy", re.M)) * HARTREE
    total_forces = -gradient(text, "CARTESIAN GRADIENT", symbols) * HARTREE / BOHR
    d4_forces = -gradient(text, "DISPERSION GRADIENT", symbols) * HARTREE / BOHR
    atoms.info.update(REF_energy=total-dispersion, total_energy=total, D4_energy=dispersion,
                      charge=formal, spin=output_spin, formal_charge_verified=True,
                      config_type="ORCA_wB97M_no_dispersion", dft_functional_basis="wB97M/def2-TZVPD",
                      source_method=route, source_file=str(path.resolve()),
                      source_sha256=hashlib.sha256(raw).hexdigest(), energy_reference="absolute",
                      dispersion_removed="ORCA_printed_D4rev", energy_units="eV", force_units="eV/Angstrom")
    atoms.arrays.update(REF_forces=total_forces-d4_forces, total_forces=total_forces, D4_forces=d4_forces)
    if not all(np.isfinite(a).all() for a in [atoms.positions, total_forces, d4_forces, [total, dispersion]]):
        raise ValueError("nonfinite values")
    return atoms


def verify(atoms):
    if atoms.info["charge"] != sum(FORMAL[s] for s in atoms.get_chemical_symbols()):
        raise ValueError("extxyz formal charge mismatch")
    np.testing.assert_allclose(atoms.info["REF_energy"] + atoms.info["D4_energy"], atoms.info["total_energy"], rtol=0, atol=1e-9)
    # ASE extxyz serializes per-atom arrays to eight decimal places.
    np.testing.assert_allclose(atoms.arrays["REF_forces"] + atoms.arrays["D4_forces"], atoms.arrays["total_forces"], rtol=0, atol=2e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputsfull/acquisition/outputs")
    parser.add_argument("--output-dir", type=Path, help="New directory only; defaults to a unique timestamped directory")
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.out"))
    if not paths:
        parser.error("no .out files found")
    destination = args.output_dir or ROOT / "outputsfull/acquisition/wb97m_nodisp" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    frames, report = [], []
    for path in paths:
        try:
            atoms = convert(path)
            verify(atoms)
        except (ValueError, AssertionError) as exc:
            report.append(dict(source=str(path.resolve()), status="rejected", reason=str(exc)))
            continue
        frames.append(atoms)
        report.append(dict(source=str(path.resolve()), status="accepted", source_sha256=atoms.info["source_sha256"], frame=len(frames)-1))
    if frames:
        target = destination / "wb97m.extxyz"
        write(target, frames, format="extxyz")
        for atoms in read(target, ":"):
            verify(atoms)
    summary = dict(input_dir=str(args.input_dir.resolve()), accepted=len(frames), rejected=len(paths)-len(frames),
                   rejection_reasons=dict(Counter(r["reason"] for r in report if r["status"] == "rejected")),
                   split="none: preserve existing source-based train/validation/test assignments", records=report)
    (destination / "report.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(f"Accepted {len(frames)}/{len(paths)}; outputs: {destination}")
    print(json.dumps(summary["rejection_reasons"], indent=2))
    if not frames:
        raise SystemExit("No valid training frames; see report.json")


if __name__ == "__main__":
    main()
