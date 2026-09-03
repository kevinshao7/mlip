#!/usr/bin/env python3
"""Prepare ORCA DFT data for target-only, naive PolarMACE fine tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read, write

from orca_to_extxyz import DEFAULT_INPUT_DIR, convert_outputs


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_TARGET_ALL = DATA_DIR / "target_all.xyz"
DEFAULT_TARGET_TRAIN = DATA_DIR / "target_train.xyz"
DEFAULT_TARGET_VALID = DATA_DIR / "target_valid.xyz"
DEFAULT_TARGET_TEST = DATA_DIR / "target_test.xyz"
DEFAULT_E0S = DATA_DIR / "target_dft_e0s.json"
DEFAULT_ATOMIZATION = MLIP_DIR / "codes" / "7_7b_clustervalidation" / "atomizationenergies.txt"


def write_e0_json(atomization_path: Path, output_path: Path) -> dict[int, float]:
    e0s: dict[int, float] = {}
    for line in atomization_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("atom"):
            continue
        symbol, value = (field.strip() for field in line.split(",", maxsplit=1))
        e0s[atomic_numbers[symbol]] = float(value)
    if not e0s:
        raise ValueError(f"No E0 values parsed from {atomization_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({str(z): v for z, v in sorted(e0s.items())}, indent=2) + "\n", encoding="utf-8")
    return e0s


def split_extxyz(input_path: Path, train_path: Path, valid_path: Path, test_path: Path, valid_fraction: float, test_fraction: float) -> dict[str, int]:
    frames = read(input_path, index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    frames = list(frames)
    if not frames:
        raise ValueError(f"No frames found in {input_path}")
    if not 0 <= valid_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Split fractions must each be in [0, 1).")
    n_total = len(frames)
    n_valid = max(1, int(round(n_total * valid_fraction))) if n_total >= 3 and valid_fraction else 0
    n_test = max(1, int(round(n_total * test_fraction))) if n_total >= 3 and test_fraction else 0
    if n_valid + n_test >= n_total:
        raise ValueError(f"Split fractions leave no training data: total={n_total}, valid={n_valid}, test={n_test}")
    valid, test, train = frames[:n_valid], frames[n_valid:n_valid + n_test], frames[n_valid + n_test:]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    write(train_path, train, format="extxyz")
    write(valid_path, valid, format="extxyz")
    write(test_path, test, format="extxyz")
    return {"train": len(train), "valid": len(valid), "test": len(test), "total": n_total}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orca-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--target-all", type=Path, default=DEFAULT_TARGET_ALL)
    parser.add_argument("--target-train", type=Path, default=DEFAULT_TARGET_TRAIN)
    parser.add_argument("--target-valid", type=Path, default=DEFAULT_TARGET_VALID)
    parser.add_argument("--target-test", type=Path, default=DEFAULT_TARGET_TEST)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--atomization-energies", type=Path, default=DEFAULT_ATOMIZATION)
    parser.add_argument("--e0s-json", type=Path, default=DEFAULT_E0S)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-orca", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.atomization_energies.is_file():
        raise FileNotFoundError(f"Missing DFT atomic-reference CSV: {args.atomization_energies}")
    e0s = write_e0_json(args.atomization_energies, args.e0s_json)
    print(f"Wrote DFT E0s for atomic numbers {sorted(e0s)} to {args.e0s_json}")
    if not args.skip_orca:
        convert_outputs(argparse.Namespace(inputs=[args.orca_dir], output=args.target_all, all_steps=False, charge=None, multiplicity=None, config_type="ORCA_DFT", allow_incomplete=args.allow_incomplete, strict=args.strict, vacuum=0.0, workers=args.workers))
    counts = split_extxyz(args.target_all, args.target_train, args.target_valid, args.target_test, args.valid_fraction, args.test_fraction)
    print("Split target data: " + " ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
