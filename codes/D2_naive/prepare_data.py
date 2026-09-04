#!/usr/bin/env python3
"""Prepare ORCA DFT data for target-only, naive PolarMACE fine tuning."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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
FORMAL_CHARGES = {1: 1, 7: -3, 8: -2}


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


def formal_charge(frame: Atoms) -> int:
    """Calculate charge using the project formal-charge convention."""
    unsupported = sorted(set(int(z) for z in frame.numbers) - FORMAL_CHARGES.keys())
    if unsupported:
        raise ValueError(f"No formal-charge rule is defined for atomic numbers {unsupported}.")
    return sum(FORMAL_CHARGES[int(z)] for z in frame.numbers)


def write_formal_charge_valid_frames(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Keep only frames whose recorded and project-formal charges agree."""
    frames = read(input_path, index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    frames = list(frames)
    valid = [frame for frame in frames if int(frame.info.get("charge", 0)) == formal_charge(frame)]
    if not valid:
        raise ValueError(f"No formal-charge-valid frames found in {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, valid, format="extxyz")
    return len(valid), len(frames) - len(valid)


def frame_stratum(frame: Atoms) -> tuple[str, int, int]:
    """Return the chemistry label used to balance the dataset splits."""
    return (
        frame.get_chemical_formula(),
        int(frame.info.get("charge", 0)),
        int(frame.info.get("spin", 1)),
    )


def apportion(
    group_sizes: dict[tuple[str, int, int], int],
    total: int,
    capacities: dict[tuple[str, int, int], int],
) -> dict[tuple[str, int, int], int]:
    """Allocate an exact total across strata using largest-remainder rounding."""
    population = sum(group_sizes.values())
    if total > sum(capacities.values()):
        raise ValueError("Requested split is larger than its available capacity.")
    quotas = {key: size * total / population for key, size in group_sizes.items()}
    allocation = {
        key: min(int(quotas[key]), capacities[key])
        for key in group_sizes
    }
    remaining = total - sum(allocation.values())
    priority = sorted(group_sizes, key=lambda key: (-(quotas[key] - int(quotas[key])), key))
    while remaining:
        assigned = 0
        for key in priority:
            if not remaining:
                break
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                assigned += 1
        if not assigned:
            break
    if remaining:
        raise RuntimeError("Could not allocate all requested split frames.")
    return allocation


def split_extxyz(input_path: Path, train_path: Path, valid_path: Path, test_path: Path, valid_fraction: float, test_fraction: float, seed: int) -> dict[str, int]:
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
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        groups[frame_stratum(frame)].append(index)
    group_sizes = {key: len(indices) for key, indices in groups.items()}

    # Allocate a representative holdout sample while retaining at least one
    # member of every chemistry stratum in training. Then divide that holdout
    # sample into validation and test sets without overlap.
    holdout_total = n_valid + n_test
    # Prefer to retain at least one example of every stratum in training. Some
    # cluster datasets contain mostly singleton formulas, making that constraint
    # incompatible with the requested holdout size; in that case allow singleton
    # strata into the holdout instead of failing dataset preparation.
    holdout_capacities = {key: max(0, size - 1) for key, size in group_sizes.items()}
    if sum(holdout_capacities.values()) < holdout_total:
        holdout_capacities = group_sizes.copy()
    holdout = apportion(group_sizes, holdout_total, holdout_capacities)
    valid_counts = apportion(holdout, n_valid, holdout)
    rng = random.Random(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []
    for key in sorted(groups):
        indices = groups[key].copy()
        rng.shuffle(indices)
        n_for_valid = valid_counts[key]
        n_for_test = holdout[key] - n_for_valid
        valid_indices.extend(indices[:n_for_valid])
        test_indices.extend(indices[n_for_valid:n_for_valid + n_for_test])
        train_indices.extend(indices[n_for_valid + n_for_test:])

    # Formula-level stratification can put no N-containing frame in a small
    # holdout when most formulas occur only once. Guarantee elemental coverage
    # for elements represented by at least three frames, using deterministic
    # train/holdout swaps that preserve split sizes and disjointness.
    for target_indices in (valid_indices, test_indices):
        for atomic_number in sorted(set(int(z) for frame in frames for z in frame.numbers)):
            containing = [index for index, frame in enumerate(frames) if atomic_number in frame.numbers]
            if len(containing) < 3 or any(atomic_number in frames[index].numbers for index in target_indices):
                continue
            donors = [index for index in train_indices if atomic_number in frames[index].numbers]
            if len(donors) < 2:
                continue
            outgoing = next(
                (index for index in target_indices if atomic_number not in frames[index].numbers),
                None,
            )
            if outgoing is None:
                continue
            incoming = donors[0]
            train_indices[train_indices.index(incoming)] = outgoing
            target_indices[target_indices.index(outgoing)] = incoming

    train = [frames[index] for index in train_indices]
    valid = [frames[index] for index in valid_indices]
    test = [frames[index] for index in test_indices]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    write(train_path, train, format="extxyz")
    write(valid_path, valid, format="extxyz")
    write(test_path, test, format="extxyz")
    return {"train": len(train), "valid": len(valid), "test": len(test), "total": n_total, "strata": len(groups)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orca-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--source-target-all",
        type=Path,
        help="Existing extxyz source to filter into --target-all; useful when raw ORCA outputs are unavailable.",
    )
    parser.add_argument("--target-all", type=Path, default=DEFAULT_TARGET_ALL)
    parser.add_argument("--target-train", type=Path, default=DEFAULT_TARGET_TRAIN)
    parser.add_argument("--target-valid", type=Path, default=DEFAULT_TARGET_VALID)
    parser.add_argument("--target-test", type=Path, default=DEFAULT_TARGET_TEST)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3, help="Random seed for reproducible stratified splits.")
    parser.add_argument("--atomization-energies", type=Path, default=DEFAULT_ATOMIZATION)
    parser.add_argument("--e0s-json", type=Path, default=DEFAULT_E0S)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-orca", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_target_all and not args.skip_orca:
        raise ValueError("--source-target-all requires --skip-orca.")
    if not args.atomization_energies.is_file():
        raise FileNotFoundError(f"Missing DFT atomic-reference CSV: {args.atomization_energies}")
    e0s = write_e0_json(args.atomization_energies, args.e0s_json)
    print(f"Wrote DFT E0s for atomic numbers {sorted(e0s)} to {args.e0s_json}")
    if not args.skip_orca:
        convert_outputs(argparse.Namespace(inputs=[args.orca_dir], output=args.target_all, all_steps=False, charge=None, multiplicity=None, config_type="ORCA_DFT", allow_incomplete=args.allow_incomplete, strict=args.strict, vacuum=0.0, workers=args.workers))
    source_target_all = args.source_target_all or args.target_all
    kept, rejected = write_formal_charge_valid_frames(source_target_all, args.target_all)
    print(f"Formal-charge validation: kept={kept} rejected={rejected}")
    counts = split_extxyz(args.target_all, args.target_train, args.target_valid, args.target_test, args.valid_fraction, args.test_fraction, args.seed)
    print("Split target data: " + " ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
