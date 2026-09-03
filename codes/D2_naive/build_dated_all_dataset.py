#!/usr/bin/env python3
"""Build a dated D2 dataset with only O-O/N-N closest-approach additions."""

from __future__ import annotations

import csv
from pathlib import Path

from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DATASET_DATE = "2026-09-03"
BASE_DATASET = SCRIPT_DIR / "data" / "target_all.xyz"
CLOSEST_DATASET = (
    MLIP_DIR / "outputsfull" / "C3_DFTproductionstopH2"
    / "processed_dft_outputs" / "target_all.extxyz"
)
SUMMARY = (
    MLIP_DIR / "outputsfull" / "C3_DFTproductionstopH2"
    / "condition_production_stopH2_clusters_summary.csv"
)
OUTPUT = SCRIPT_DIR / "data" / f"target_all_{DATASET_DATE}_ON_closest_approach.xyz"
ALLOWED_PAIRS = {"O2": "O-O", "N2": "N-N"}


def main() -> None:
    base = read(BASE_DATASET, index=":")
    closest = read(CLOSEST_DATASET, index=":")
    with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        provenance = {
            int(row["cluster_id"]) - 1: row for row in csv.DictReader(handle)
        }

    additions = []
    for atoms in closest:
        source_frame = int(atoms.info["source_frame"])
        row = provenance[source_frame]
        molecule = row["molecule"]
        if molecule not in ALLOWED_PAIRS or row["sample_kind"] != "near_formation":
            continue
        atoms.info.update({
            "dataset_date": DATASET_DATE,
            "candidate_pair": ALLOWED_PAIRS[molecule],
            "event": "closest_approach",
            "selection_class": "near_formation_not_formed",
            "source_condition": row["condition"],
            "seed_distance_A": float(row["seed_distance_A"]),
        })
        additions.append(atoms)

    for atoms in base:
        atoms.info["dataset_date"] = DATASET_DATE
        atoms.info["dataset_role"] = "D2_baseline"
    for atoms in additions:
        atoms.info["dataset_role"] = "O_O_or_N_N_closest_approach"

    write(OUTPUT, base + additions, format="extxyz")
    pair_counts = {
        pair: sum(a.info["candidate_pair"] == pair for a in additions)
        for pair in sorted(ALLOWED_PAIRS.values())
    }
    print(f"wrote {OUTPUT}")
    print(f"baseline={len(base)} additions={len(additions)} total={len(base) + len(additions)}")
    print("closest-approach additions:", pair_counts)


if __name__ == "__main__":
    main()
