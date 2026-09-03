#!/usr/bin/env python3
"""Apply the B-production species graph analysis to every D2 target frame."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "B_conditionsproduction"
sys.path.insert(0, str(BASE_DIR))

from plot_final_species_distribution import Frame, component_formula, molecular_components  # noqa: E402


INPUT = SCRIPT_DIR / "data" / "target_all.xyz"
OUTPUT_DIR = SCRIPT_DIR / "data" / "target_species_distribution"
BOND_SCALE = 1.20


def nonperiodic_frame(atoms) -> Frame:
    """Represent a nonperiodic configuration in a box too large for MIC wrapping."""
    positions = np.asarray(atoms.positions, dtype=float)
    spans = np.ptp(positions, axis=0)
    box_lengths = 2.0 * spans + 20.0
    return Frame(
        tuple(atoms.get_chemical_symbols()),
        positions,
        np.diag(box_lengths),
        str(atoms.info),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_species: Counter[str] = Counter()
    frame_rows: list[dict[str, object]] = []

    for frame_index, atoms in enumerate(iread(INPUT, index=":")):
        frame = nonperiodic_frame(atoms)
        components, minimum_distance = molecular_components(
            frame,
            oh_cutoff=1.45,
            nh_cutoff=1.30,
            hh_cutoff=0.5,
            bond_scale=BOND_SCALE,
        )
        species = Counter(component_formula(frame, component) for component in components)
        total_species.update(species)
        frame_rows.append(
            {
                "frame_index_0based": frame_index,
                "source_file": atoms.info.get("source_file", ""),
                "charge": atoms.info.get("charge", ""),
                "natoms": len(atoms),
                "component_count": len(components),
                "largest_component_atoms": max(map(len, components)),
                "minimum_interatomic_distance_A": minimum_distance,
                "molecular_composition": ":".join(
                    f"{name}({count})" for name, count in species.most_common()
                ),
            }
        )

    if not frame_rows:
        raise ValueError(f"No configurations found in {INPUT}")

    with (OUTPUT_DIR / "species_by_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    summary = {
        "input": str(INPUT),
        "frame_count": len(frame_rows),
        "distance_mode": "nonperiodic",
        "bond_scale": BOND_SCALE,
        "hydrogen_assignment": "nearest_H_O_or_N",
        "count_definition": "sum of connected-component occurrences over all independent frames",
        "total_components": sum(total_species.values()),
        "species": dict(total_species.most_common()),
    }
    with (OUTPUT_DIR / "species_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    ordered = total_species.most_common()
    names = [name for name, _count in ordered]
    counts = [count for _name, count in ordered]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(names)), 5.5), constrained_layout=True)
    bars = ax.bar(np.arange(len(names)), counts, color="#3b82a0", edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(np.arange(len(names)), names, rotation=55, ha="right")
    ax.set_ylabel("Component occurrences across all configurations")
    ax.set_title("D2 target_all.xyz species distribution\nEach H assigned to nearest H/O/N")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUTPUT_DIR / "species_distribution.png", dpi=220)
    plt.close(fig)

    print(f"Analysed {len(frame_rows)} frames and {sum(total_species.values())} components")
    print(f"Species: {dict(total_species.most_common())}")
    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
