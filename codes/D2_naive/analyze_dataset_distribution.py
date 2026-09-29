#!/usr/bin/env python3
"""Audit D2 extxyz split composition, validity, overlap, and balance.

This is deliberately read-only with respect to the input datasets.  It writes
an auditable JSON summary and a compact CSV table, and exits nonzero when the
split files are not a disjoint partition of the supplied all-data file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import read


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FORMAL_CHARGES = {1: 1, 7: -3, 8: -2}
SPLIT_NAMES = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-file", type=Path, default=DATA_DIR / "target_all.xyz")
    for split in SPLIT_NAMES:
        parser.add_argument(f"--{split}-file", type=Path, default=DATA_DIR / f"target_{split}.xyz")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "dataset_distribution")
    parser.add_argument("--position-decimals", type=int, default=8,
                        help="Decimal places used only for duplicate-geometry fingerprints.")
    return parser.parse_args()


def load_frames(path: Path) -> list[Atoms]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames = read(path, index=":")
    return [frames] if isinstance(frames, Atoms) else list(frames)


def formal_charge(frame: Atoms) -> int | None:
    numbers = set(map(int, frame.numbers))
    if not numbers.issubset(FORMAL_CHARGES):
        return None
    return sum(FORMAL_CHARGES[int(number)] for number in frame.numbers)


def frame_fingerprint(frame: Atoms, decimals: int) -> str:
    """Fingerprint geometry, cell, PBC, and declared charge/spin; not labels."""
    payload = {
        "numbers": list(map(int, frame.numbers)),
        "positions": np.round(frame.positions, decimals).tolist(),
        "cell": np.round(frame.cell.array, decimals).tolist(),
        "pbc": list(map(bool, frame.pbc)),
        "charge": int(frame.info.get("charge", 0)),
        "spin": int(frame.info.get("spin", 1)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def size_bin(natoms: int) -> str:
    lower = (natoms // 10) * 10
    return f"{lower}-{lower + 9}"


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def describe(frames: Iterable[Atoms]) -> dict[str, object]:
    frames = list(frames)
    formulas = Counter(frame.get_chemical_formula() for frame in frames)
    charges = Counter(int(frame.info.get("charge", 0)) for frame in frames)
    spins = Counter(int(frame.info.get("spin", 1)) for frame in frames)
    sources = Counter(str(frame.info.get("source_dataset", "unspecified")) for frame in frames)
    atom_counts = Counter()
    frame_element_presence = Counter()
    size_bins = Counter()
    formal_mismatches: list[int] = []
    unsupported_formal_charge: list[int] = []
    for index, frame in enumerate(frames):
        symbols = frame.get_chemical_symbols()
        atom_counts.update(symbols)
        frame_element_presence.update(set(symbols))
        size_bins[size_bin(len(frame))] += 1
        expected = formal_charge(frame)
        if expected is None:
            unsupported_formal_charge.append(index)
        elif int(frame.info.get("charge", 0)) != expected:
            formal_mismatches.append(index)
    return {
        "frame_count": len(frames),
        "atom_count": sum(atom_counts.values()),
        "atoms_by_element": counter_dict(atom_counts),
        "frames_containing_element": counter_dict(frame_element_presence),
        "formulas": counter_dict(formulas),
        "charges": counter_dict(charges),
        "spins": counter_dict(spins),
        "source_datasets": counter_dict(sources),
        "natoms_bins": counter_dict(size_bins),
        "formal_charge_mismatch_indices": formal_mismatches,
        "unsupported_formal_charge_indices": unsupported_formal_charge,
    }


def duplicate_indices(fingerprints: list[str]) -> list[list[int]]:
    locations: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(fingerprints):
        locations.setdefault(fingerprint, []).append(index)
    return [indices for indices in locations.values() if len(indices) > 1]


def write_csv(path: Path, summaries: dict[str, dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for split, summary in summaries.items():
        total = int(summary["frame_count"])
        for element, count in summary["frames_containing_element"].items():
            rows.append({"split": split, "metric": "frames_containing_element", "category": element,
                         "count": count, "fraction_of_frames": count / total if total else 0})
        for element, count in summary["atoms_by_element"].items():
            rows.append({"split": split, "metric": "atoms_by_element", "category": element,
                         "count": count, "fraction_of_frames": ""})
        for source, count in summary["source_datasets"].items():
            rows.append({"split": split, "metric": "source_dataset", "category": source,
                         "count": count, "fraction_of_frames": count / total if total else 0})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "metric", "category", "count", "fraction_of_frames"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split_paths = {name: getattr(args, f"{name}_file") for name in SPLIT_NAMES}
    all_frames = load_frames(args.all_file)
    split_frames = {name: load_frames(path) for name, path in split_paths.items()}
    all_fingerprints = [frame_fingerprint(frame, args.position_decimals) for frame in all_frames]
    split_fingerprints = {
        name: [frame_fingerprint(frame, args.position_decimals) for frame in frames]
        for name, frames in split_frames.items()
    }
    split_sets = {name: set(fingerprints) for name, fingerprints in split_fingerprints.items()}
    pair_overlaps = {
        f"{left}_{right}": len(split_sets[left] & split_sets[right])
        for index, left in enumerate(SPLIT_NAMES) for right in SPLIT_NAMES[index + 1:]
    }
    partition_set = set().union(*split_sets.values())
    all_set = set(all_fingerprints)
    summary = {
        "inputs": {"all": str(args.all_file), **{name: str(path) for name, path in split_paths.items()}},
        "fingerprint_position_decimals": args.position_decimals,
        "all": describe(all_frames),
        "splits": {name: describe(frames) for name, frames in split_frames.items()},
        "integrity": {
            "duplicate_geometries_within_all": duplicate_indices(all_fingerprints),
            "duplicate_geometries_within_splits": {
                name: duplicate_indices(fingerprints) for name, fingerprints in split_fingerprints.items()
            },
            "cross_split_geometry_overlap": pair_overlaps,
            "split_union_missing_from_all": len(partition_set - all_set),
            "all_frames_missing_from_split_union": len(all_set - partition_set),
            "split_frame_count": sum(len(frames) for frames in split_frames.values()),
            "all_frame_count": len(all_frames),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dataset_distribution_summary.json"
    csv_path = args.output_dir / "dataset_distribution_by_split.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, {"all": summary["all"], **summary["splits"]})
    integrity = summary["integrity"]
    bad = any(pair_overlaps.values()) or integrity["split_union_missing_from_all"] or integrity["all_frames_missing_from_split_union"]
    print(f"Wrote {json_path} and {csv_path}")
    print("Split frame counts: " + ", ".join(f"{name}={len(frames)}" for name, frames in split_frames.items()))
    print("Cross-split geometry overlap: " + ", ".join(f"{name}={count}" for name, count in pair_overlaps.items()))
    if bad:
        raise SystemExit("Dataset split integrity check failed; see JSON summary.")


if __name__ == "__main__":
    main()
