#!/usr/bin/env python3
"""Create a compact OVITO trajectory for auditing molecular-species assignments.

Each output frame contains one representative target component and the four
nearest complete components.  Component construction is imported from
plot_final_species_distribution.py so this audit uses exactly the same PBC,
heavy-atom cutoff, and nearest-H assignment as the plots.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.geometry import find_mic

import plot_final_species_distribution as species_base
import plot_species_time_series as time_series


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "speciestest.xyz"


@dataclass
class Example:
    run_id: str
    frame_index: int
    frame: species_base.Frame
    target: list[int]
    components: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=species_base.DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=species_base.DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-id", action="append",
        help="Run to scan; repeat as needed. Default: all 100 GPa manifest runs.",
    )
    parser.add_argument("--pressure-gpa", type=float, default=100.0)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4, help="Parallel trajectory workers (default: 4).")
    parser.add_argument(
        "--max-examples", type=int, default=20,
        help="Maximum audit frames (default: 20, balanced across selected runs).",
    )
    parser.add_argument("--oh-cutoff", type=float, default=1.45)
    parser.add_argument("--nh-cutoff", type=float, default=1.30)
    parser.add_argument("--hh-cutoff", type=float, default=0.50)
    parser.add_argument("--bond-scale", type=float, default=1.20)
    return parser.parse_args()


def component_distance(frame: species_base.Frame, first: list[int], second: list[int]) -> float:
    distances = species_base.minimum_image_distances(
        frame.positions[first], frame.positions[second], frame.cell
    )
    return float(distances.min())


def molecular_components_fast(
    frame: species_base.Frame,
    oh_cutoff: float,
    nh_cutoff: float,
    hh_cutoff: float,
    bond_scale: float,
) -> list[list[int]]:
    """Vectorized equivalent of species_base.molecular_components()."""
    natoms = len(frame.symbols)
    displacement = frame.positions[:, None, :] - frame.positions[None, :, :]
    _vectors, flat_distances = find_mic(
        displacement.reshape(-1, 3), cell=frame.cell, pbc=True
    )
    distances = flat_distances.reshape(natoms, natoms)
    adjacency = [set() for _ in range(natoms)]
    for left in range(natoms - 1):
        if frame.symbols[left] == "H":
            continue
        for right in range(left + 1, natoms):
            if frame.symbols[right] == "H":
                continue
            cutoff = species_base.pair_cutoff(
                frame.symbols[left], frame.symbols[right],
                oh_cutoff, nh_cutoff, hh_cutoff, bond_scale,
            )
            if distances[left, right] <= cutoff:
                adjacency[left].add(right)
                adjacency[right].add(left)
    eligible = np.array(
        [index for index, symbol in enumerate(frame.symbols) if symbol in {"H", "O", "N"}],
        dtype=int,
    )
    for hydrogen, symbol in enumerate(frame.symbols):
        if symbol != "H":
            continue
        candidates = eligible[eligible != hydrogen]
        if candidates.size:
            nearest = int(candidates[int(np.argmin(distances[hydrogen, candidates]))])
            adjacency[hydrogen].add(nearest)
            adjacency[nearest].add(hydrogen)
    unseen = set(range(natoms))
    components: list[list[int]] = []
    while unseen:
        start = unseen.pop()
        component = [start]
        stack = [start]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def unwrapped_subset_positions(
    frame: species_base.Frame, component_indices: list[list[int]], target: list[int]
) -> dict[int, np.ndarray]:
    """Put the small audit cluster together across periodic boundaries."""
    anchor = target[0]
    anchor_position = frame.positions[anchor]
    atom_indices = [index for component in component_indices for index in component]
    displacement = frame.positions[atom_indices] - anchor_position
    mic_vectors, _distances = find_mic(displacement, cell=frame.cell, pbc=True)
    center = 0.5 * np.sum(frame.cell, axis=0)
    return {
        index: center + vector for index, vector in zip(atom_indices, mic_vectors, strict=True)
    }


def write_example(handle, example: Example, neighbor_count: int) -> None:
    frame = example.frame
    target_formula = species_base.component_formula(frame, example.target)
    target_category = time_series.species_category(target_formula)
    others = [component for component in example.components if component != example.target]
    others.sort(key=lambda component: component_distance(frame, example.target, component))
    neighbors = others[:neighbor_count]
    shown = [example.target, *neighbors]
    positions = unwrapped_subset_positions(frame, shown, example.target)
    atom_count = sum(map(len, shown))
    lattice = " ".join(f"{value:.10g}" for value in frame.cell.reshape(-1))
    neighbor_formulas = ",".join(
        species_base.component_formula(frame, component) for component in neighbors
    )
    target_ids = ",".join(map(str, example.target))
    comment = (
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:source_atom_id:I:1:'
        f'component_id:I:1:molecule:S:1 target_species="{target_formula}" '
        f'run_id="{example.run_id}" condensed_frame_index_0based={example.frame_index} '
        f'target_source_atom_ids="{target_ids}" neighbor_species="{neighbor_formulas}" pbc="T T T"'
        f' target_category="{target_category}"'
    )
    handle.write(f"{atom_count}\n{comment}\n")
    for component_id, component in enumerate(shown):
        molecule = species_base.component_formula(frame, component)
        for atom_index in component:
            x, y, z = positions[atom_index]
            handle.write(
                f"{frame.symbols[atom_index]} {x:.10f} {y:.10f} {z:.10f} "
                f"{atom_index} {component_id} {molecule}\n"
            )


def collect_run_examples(
    task: tuple[str, str, int, float, float, float, float],
) -> tuple[str, list[Example]]:
    run_id, trajectory_string, limit, oh_cutoff, nh_cutoff, hh_cutoff, bond_scale = task
    trajectory = Path(trajectory_string)
    print(f"Scanning {run_id}: {trajectory}", flush=True)
    examples: dict[str, Example] = {}
    for frame_index, frame in species_base.iter_xyz_frames(trajectory):
        components = molecular_components_fast(
            frame, oh_cutoff, nh_cutoff, hh_cutoff, bond_scale
        )
        for component in components:
            formula = species_base.component_formula(frame, component)
            category = time_series.species_category(formula)
            if category not in {"unexpected_cluster", "improper_species"}:
                continue
            examples.setdefault(
                formula, Example(run_id, frame_index, frame, component, components)
            )
            if len(examples) >= limit:
                break
        if len(examples) >= limit:
            break
    print(f"  {run_id}: collected {len(examples)} audit species", flush=True)
    return run_id, list(examples.values())


def main() -> None:
    args = parse_args()
    if (
        args.neighbors < 0
        or args.max_examples < 1
        or args.workers < 1
        or min(args.oh_cutoff, args.nh_cutoff, args.hh_cutoff, args.bond_scale) <= 0
    ):
        raise ValueError("Counts must be valid and cutoffs/scales must be positive")
    rows = species_base.manifest_rows(args.manifest.resolve())
    if args.run_id:
        requested = set(args.run_id)
        rows = [row for row in rows if row["run_id"] in requested]
        missing = requested - {row["run_id"] for row in rows}
        if missing:
            raise ValueError(f"Run IDs absent from manifest: {', '.join(sorted(missing))}")
    else:
        rows = [row for row in rows if np.isclose(float(row["pressure_GPa"]), args.pressure_gpa)]
    if not rows:
        raise ValueError("No trajectories selected")

    per_run_limit = max(1, int(np.ceil(args.max_examples / len(rows))))
    tasks = [
        (
            row["run_id"],
            str(species_base.trajectory_for_run(args.input_dir.resolve(), row["run_id"])),
            per_run_limit,
            args.oh_cutoff,
            args.nh_cutoff,
            args.hh_cutoff,
            args.bond_scale,
        )
        for row in rows
    ]
    effective_workers = min(args.workers, len(tasks))
    print(f"Scanning {len(tasks)} runs with {effective_workers} worker processes", flush=True)
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        collected = dict(executor.map(collect_run_examples, tasks))
    selected_examples = [
        example for row in rows for example in collected[row["run_id"]]
    ][: args.max_examples]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for example in selected_examples:
            write_example(handle, example, args.neighbors)
    print(f"Wrote {len(selected_examples)} audit frames to {output}")


if __name__ == "__main__":
    main()
