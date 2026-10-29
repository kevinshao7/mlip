#!/usr/bin/env python3
"""Extract four molecule-plus-five-neighbour clusters near 20 GPa."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import find_mic
from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = MLIP_DIR / "outputsfull" / "B2a_conditionsfinereduced"
DEFAULT_OUTPUT = SCRIPT_DIR / "clusters_20GPa_pilot4.xyz"
SPECIES_CODE = MLIP_DIR / "codes" / "B_conditionsproduction" / "plot_final_species_distribution.py"
DEFAULT_CONDITION = "P17p7828GPa_R0p5"
DEFAULT_FRAME_FRACTIONS = (0.20, 0.40, 0.60, 0.80)
FORMAL_CHARGES = {"H": 1, "N": -3, "O": -2, "S": -2}


def load_species_code():
    spec = importlib.util.spec_from_file_location("c4_species_definition", SPECIES_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load molecular definition from {SPECIES_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


species = load_species_code()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument(
        "--frame-fractions", type=float, nargs=4, default=DEFAULT_FRAME_FRACTIONS,
        metavar=("F1", "F2", "F3", "F4"),
        help="Four fractions through the trajectory (default: 0.20 0.40 0.60 0.80).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--oh-cutoff", type=float, default=1.45)
    parser.add_argument("--nh-cutoff", type=float, default=1.30)
    parser.add_argument("--hh-cutoff", type=float, default=0.50)
    parser.add_argument("--bond-scale", type=float, default=1.20)
    return parser.parse_args()


def trajectory_for_condition(root: Path, condition: str) -> Path:
    matches = sorted((root / condition).glob("production*.xyz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one production*.xyz in {root / condition}, found {len(matches)}")
    return matches[0]


def frame_count_from_manifest(root: Path, condition: str) -> int:
    manifest = root / "reduction_manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    if condition not in rows:
        raise KeyError(f"Condition {condition} is absent from {manifest}")
    count = int(rows[condition]["kept_frames"])
    if count < 1:
        raise ValueError(f"Manifest reports no frames for {condition}")
    return count


def read_frame(path: Path, index: int):
    for frame_index, frame in species.iter_xyz_frames(path):
        if frame_index == index:
            return frame
    raise IndexError(f"Frame {index} not found in {path}")


def read_frames(path: Path, indices: list[int]) -> dict[int, object]:
    wanted = set(indices)
    found = {}
    for frame_index, frame in species.iter_xyz_frames(path):
        if frame_index in wanted:
            found[frame_index] = frame
        if len(found) == len(wanted):
            break
    missing = wanted - found.keys()
    if missing:
        raise IndexError(f"Frames absent from {path}: {sorted(missing)}")
    return found


def component_distance(frame, first: list[int], second: list[int]) -> float:
    delta = frame.positions[np.asarray(first)][:, None, :] - frame.positions[np.asarray(second)][None, :, :]
    _vectors, distances = find_mic(delta.reshape(-1, 3), cell=frame.cell, pbc=True)
    return float(np.min(distances))


def charge_and_spin(symbols: list[str]) -> tuple[int, int]:
    missing = sorted(set(symbols) - FORMAL_CHARGES.keys())
    if missing:
        raise ValueError(f"No formal-charge rule for elements: {', '.join(missing)}")
    charge = sum(FORMAL_CHARGES[symbol] for symbol in symbols)
    electrons = sum(int(atomic_numbers[symbol]) for symbol in symbols) - charge
    return charge, 2 if electrons % 2 else 1


def make_cluster(frame, components: list[list[int]], target_index: int, vacuum: float) -> Atoms:
    target = components[target_index]
    ranked = sorted(
        ((component_distance(frame, target, component), index)
         for index, component in enumerate(components) if index != target_index),
        key=lambda item: (item[0], item[1]),
    )
    if len(ranked) < 5:
        raise ValueError("Frame contains fewer than six molecular components")
    neighbours = ranked[:5]
    selected_components = [target] + [components[index] for _distance, index in neighbours]
    selected = [atom for component in selected_components for atom in component]

    anchor = frame.positions[target[0]]
    delta = frame.positions[np.asarray(selected)] - anchor
    unwrapped, _distances = find_mic(delta, cell=frame.cell, pbc=True)
    positions = unwrapped + vacuum / 2.0
    symbols = [frame.symbols[index] for index in selected]
    cluster = Atoms(symbols=symbols, positions=positions, cell=[vacuum] * 3, pbc=False)
    charge, spin = charge_and_spin(symbols)
    cluster.info.update({
        "charge": charge,
        "spin": spin,
        "target_component_index": target_index,
        "target_formula": species.component_formula(frame, target),
        "target_atom_indices": json.dumps(target),
        "neighbour_component_indices": json.dumps([index for _distance, index in neighbours]),
        "neighbour_formulas": json.dumps([species.component_formula(frame, components[index]) for _distance, index in neighbours]),
        "neighbour_min_distances_A": json.dumps([round(distance, 10) for distance, _index in neighbours]),
        "cluster_rule": "target_molecule_plus_5_nearest_molecules_by_minimum_interatomic_MIC_distance",
        "molecule_definition": "B_conditionsproduction/species_time_series_nearest_H_assignment",
    })
    return cluster


def main() -> None:
    args = parse_args()
    if any(not 0.0 <= fraction <= 1.0 for fraction in args.frame_fractions) or args.vacuum <= 0:
        raise ValueError("Every --frame-fractions value must be in [0,1] and --vacuum must be positive")

    condition = args.condition
    source = trajectory_for_condition(args.input_root, condition)
    n_frames = frame_count_from_manifest(args.input_root, condition)
    frame_indices = [
        min(n_frames - 1, int(round(fraction * (n_frames - 1))))
        for fraction in args.frame_fractions
    ]
    if len(set(frame_indices)) != 4:
        raise ValueError(f"Frame fractions do not select four distinct frames: {frame_indices}")
    frames = read_frames(source, frame_indices)

    clusters: list[Atoms] = []
    rows: list[dict[str, object]] = []
    for pilot_index, frame_index in enumerate(frame_indices):
        frame = frames[frame_index]
        components, minimum_distance = species.molecular_components(
            frame, args.oh_cutoff, args.nh_cutoff, args.hh_cutoff, args.bond_scale
        )
        # A per-condition generator makes selection reproducible and independent of processing order.
        rng = np.random.default_rng(args.seed + pilot_index)
        target_index = int(rng.integers(len(components)))
        cluster = make_cluster(frame, components, target_index, args.vacuum)
        cluster.info.update({
            "cluster_id": pilot_index + 1,
            "condition": condition,
            "source_xyz": str(source),
            "source_reduced_frame": frame_index,
            "source_frame_fraction": frame_index / max(n_frames - 1, 1),
            "source_comment": frame.comment,
            "frame_minimum_interatomic_distance_A": minimum_distance,
            "selection_seed": args.seed + pilot_index,
            "oh_cutoff_A": args.oh_cutoff,
            "nh_cutoff_A": args.nh_cutoff,
            "hh_cutoff_A": args.hh_cutoff,
            "heavy_atom_bond_scale": args.bond_scale,
        })
        clusters.append(cluster)
        rows.append({
            "cluster_id": pilot_index + 1, "condition": condition,
            "source_reduced_frame": frame_index, "target_component_index": target_index,
            "target_formula": cluster.info["target_formula"], "natoms": len(cluster),
            "charge": cluster.info["charge"], "spin": cluster.info["spin"],
            "neighbour_formulas": cluster.info["neighbour_formulas"],
            "neighbour_min_distances_A": cluster.info["neighbour_min_distances_A"],
        })
        print(f"cluster {pilot_index + 1}: {condition} frame={frame_index} "
              f"target={cluster.info['target_formula']} atoms={len(cluster)} "
              f"charge={cluster.info['charge']} spin={cluster.info['spin']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, clusters, format="extxyz")
    summary = args.output.with_suffix(".csv")
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} and {summary}")


if __name__ == "__main__":
    main()
