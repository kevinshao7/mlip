#!/usr/bin/env python3
"""Analyse molecular species in production trajectories.

Every condensed stride-100 frame is scanned for N/O components without H and
components containing only H. Final-frame species plots are also produced.
Independent conditions are processed by eight worker processes by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Prevent each worker from starting its own native thread pool.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import find_mic
from ase.data import atomic_numbers, covalent_radii


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_DIR = MLIP_DIR / "outputsfull" / "B1_conditionsproduction_stride100_xyz"
DEFAULT_MANIFEST = SCRIPT_DIR / "production_manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "final_species_distribution"
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
CONDENSED_OUTPUT_STRIDE = 100
TARGET_PRESSURE_GPA = 100.0


@dataclass(frozen=True)
class Frame:
    symbols: tuple[str, ...]
    positions: np.ndarray
    cell: np.ndarray
    comment: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-id", action="append", help="Run ID to analyse; repeat to select several.")
    selection.add_argument("--all-conditions", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes (default: 8).")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print all-frame scan progress every N condensed frames (default: 10).",
    )
    parser.add_argument("--oh-cutoff", type=float, default=1.45, help="Legacy option; H bonds now use nearest-atom assignment.")
    parser.add_argument("--nh-cutoff", type=float, default=1.30, help="Legacy option; H bonds now use nearest-atom assignment.")
    parser.add_argument("--hh-cutoff", type=float, default=0.5, help="Legacy option; H bonds now use nearest-atom assignment.")
    parser.add_argument(
        "--bond-scale",
        type=float,
        default=1.20,
        help="Covalent-radius multiplier for all other element pairs.",
    )
    parser.add_argument(
        "--min-species-count",
        type=int,
        default=1,
        help="Only label species occurring at least this many times in plots.",
    )
    return parser.parse_args()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Production manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "run_id" not in rows[0]:
        raise ValueError(f"Manifest has no run_id records: {path}")
    return rows


def selected_rows(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if args.all_conditions:
        selected = rows
    elif not args.run_id:
        selected = rows
    else:
        requested = set(args.run_id)
        selected = [row for row in rows if row["run_id"] in requested]
        missing = requested - {row["run_id"] for row in selected}
        if missing:
            raise ValueError(f"Run IDs absent from manifest: {', '.join(sorted(missing))}")

    selected = [
        row for row in selected if np.isclose(float(row["pressure_GPa"]), TARGET_PRESSURE_GPA)
    ]
    if not selected:
        raise ValueError(f"No selected manifest conditions are at {TARGET_PRESSURE_GPA:g} GPa")
    return selected


def trajectory_for_run(input_dir: Path, run_id: str) -> Path:
    folder = input_dir / run_id
    matches = sorted(folder.glob("*.xyz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one XYZ trajectory in {folder}, found {len(matches)}")
    return matches[0]


def tail_nonempty_lines(path: Path, count: int) -> list[str]:
    """Read the last count non-empty lines without scanning a multi-GB trajectory."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        data = b""
        while position > 0 and len(data.splitlines()) < count + 2:
            size = min(1024 * 1024, position)
            position -= size
            handle.seek(position)
            data = handle.read(size) + data
    lines = [line.decode("utf-8", errors="replace").strip() for line in data.splitlines() if line.strip()]
    return lines[-count:]


def read_last_xyz_frame(path: Path) -> Frame:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
    try:
        natoms = int(first)
    except ValueError as exc:
        raise ValueError(f"First line of {path} is not an XYZ atom count: {first!r}") from exc

    lines = tail_nonempty_lines(path, natoms + 2)
    if len(lines) != natoms + 2 or int(lines[0]) != natoms:
        raise ValueError(f"Could not isolate a complete {natoms}-atom final frame in {path}")
    comment = lines[1]
    lattice_match = LATTICE_RE.search(comment)
    if not lattice_match:
        raise ValueError(f"Final frame in {path} has no Lattice field")
    lattice = np.fromstring(lattice_match.group(1), sep=" ")
    if lattice.size != 9:
        raise ValueError(f"Final frame in {path} has an invalid Lattice field")

    symbols: list[str] = []
    positions = np.empty((natoms, 3), dtype=float)
    for index, line in enumerate(lines[2:]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed atom line {index + 1} in final frame of {path}")
        symbols.append(fields[0])
        positions[index] = [float(value) for value in fields[1:4]]
    return Frame(tuple(symbols), positions, lattice.reshape(3, 3), comment)


def iter_xyz_frames(path: Path):
    """Yield every frame in an XYZ trajectory without loading it all into memory."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        while True:
            atom_count_line = handle.readline()
            while atom_count_line and not atom_count_line.strip():
                atom_count_line = handle.readline()
            if not atom_count_line:
                return
            try:
                natoms = int(atom_count_line.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid atom count at condensed frame {frame_index} in {path}: "
                    f"{atom_count_line.strip()!r}"
                ) from exc

            comment = handle.readline().strip()
            lattice_match = LATTICE_RE.search(comment)
            if not lattice_match:
                raise ValueError(f"Condensed frame {frame_index} in {path} has no Lattice field")
            lattice = np.fromstring(lattice_match.group(1), sep=" ")
            if lattice.size != 9:
                raise ValueError(f"Condensed frame {frame_index} in {path} has an invalid Lattice field")

            symbols: list[str] = []
            positions = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                line = handle.readline()
                if not line:
                    raise ValueError(f"Incomplete condensed frame {frame_index} in {path}")
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(
                        f"Malformed atom {atom_index} in condensed frame {frame_index} of {path}"
                    )
                symbols.append(fields[0])
                positions[atom_index] = [float(value) for value in fields[1:4]]
            yield frame_index, Frame(tuple(symbols), positions, lattice.reshape(3, 3), comment)
            frame_index += 1


def minimum_image_distances(
    first_positions: np.ndarray,
    second_positions: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    delta = first_positions[:, None, :] - second_positions[None, :, :]
    shape = delta.shape[:2]
    _minimum_image_vectors, distances = find_mic(
        delta.reshape(-1, 3),
        cell=cell,
        pbc=True,
    )
    return distances.reshape(shape)


def pair_cutoff(
    first: str,
    second: str,
    oh_cutoff: float,
    nh_cutoff: float,
    hh_cutoff: float,
    bond_scale: float,
) -> float:
    pair = frozenset((first, second))
    if pair == {"H", "O"}:
        return oh_cutoff
    if pair == {"H", "N"}:
        return nh_cutoff
    if first == second == "H":
        return hh_cutoff
    return bond_scale * (
        covalent_radii[atomic_numbers[first]] + covalent_radii[atomic_numbers[second]]
    )


def molecular_components(
    frame: Frame,
    oh_cutoff: float,
    nh_cutoff: float,
    hh_cutoff: float,
    bond_scale: float,
) -> tuple[list[list[int]], float]:
    """Build molecules by assigning every H to its nearest H, O, or N atom.

    Heavy-atom bonds are still identified with the covalent-radius criterion.
    Hydrogen assignments are unconditional and therefore do not use the H-H,
    O-H, or N-H cutoffs; each hydrogen contributes exactly one assigned edge.
    """
    natoms = len(frame.symbols)
    adjacency = [set() for _ in range(natoms)]
    minimum_distance = np.inf

    # Add bonds between non-hydrogen atoms using the usual distance criterion.
    for left in range(natoms - 1):
        distances = minimum_image_distances(
            frame.positions[left : left + 1], frame.positions[left + 1 :], frame.cell
        )[0]
        minimum_distance = min(minimum_distance, float(distances.min()))
        for offset, distance in enumerate(distances, start=left + 1):
            if "H" in (frame.symbols[left], frame.symbols[offset]):
                continue
            if distance <= pair_cutoff(
                frame.symbols[left], frame.symbols[offset],
                oh_cutoff, nh_cutoff, hh_cutoff, bond_scale,
            ):
                adjacency[left].add(offset)
                adjacency[offset].add(left)

    # Assign each hydrogen to one nearest eligible atom.  Turning these
    # directed assignments into undirected graph edges naturally handles H-H
    # pairs and chains while preserving the one-assignment-per-H rule.
    eligible = np.array(
        [index for index, symbol in enumerate(frame.symbols) if symbol in {"H", "O", "N"}],
        dtype=int,
    )
    for hydrogen in (index for index, symbol in enumerate(frame.symbols) if symbol == "H"):
        candidates = eligible[eligible != hydrogen]
        if candidates.size == 0:
            continue
        distances = minimum_image_distances(
            frame.positions[hydrogen : hydrogen + 1],
            frame.positions[candidates],
            frame.cell,
        )[0]
        nearest = int(candidates[int(np.argmin(distances))])
        adjacency[hydrogen].add(nearest)
        adjacency[nearest].add(hydrogen)

    unseen = set(range(natoms))
    components: list[list[int]] = []
    while unseen:
        start = unseen.pop()
        component = [start]
        stack = [start]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.append(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components, float(minimum_distance)


def component_formula(frame: Frame, component: list[int]) -> str:
    counts = Counter(frame.symbols[index] for index in component)
    order = ["C"] + (["H"] if "H" in counts else []) + sorted(counts.keys() - {"C", "H"}) if "C" in counts else sorted(counts)
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] > 1 else "") for symbol in order)


def flagged_component_event(
    row: dict[str, str], trajectory: Path, frame_index: int, frame: Frame, component: list[int]
) -> dict[str, object] | None:
    counts = Counter(frame.symbols[index] for index in component)
    if counts["H"] == 0 and (counts["N"] > 0 or counts["O"] > 0):
        event_type = "N_or_O_component_without_H"
    elif counts["H"] == len(component):
        event_type = "H_only_component"
    else:
        return None
    return {
        "run_id": row["run_id"],
        "pressure_GPa": float(row["pressure_GPa"]),
        "temperature_K": float(row["temperature_K"]),
        "ammonia_water_ratio": float(row["ammonia_water_ratio"]),
        "condensed_frame_index_0based": frame_index,
        "estimated_original_frame_index_0based": frame_index * CONDENSED_OUTPUT_STRIDE,
        "condensed_output_stride": CONDENSED_OUTPUT_STRIDE,
        "event_type": event_type,
        "species": component_formula(frame, component),
        "component_size": len(component),
        "atom_ids_0based": ";".join(map(str, component)),
        "H_count": counts["H"],
        "N_count": counts["N"],
        "O_count": counts["O"],
        "trajectory": str(trajectory),
    }


def analyse_run(task: tuple[dict[str, str], str, float, float, float, float, int]) -> dict[str, object]:
    row, trajectory_string, oh_cutoff, nh_cutoff, hh_cutoff, bond_scale, progress_every = task
    trajectory = Path(trajectory_string)
    flagged_events: list[dict[str, object]] = []
    frame = None
    frame_count = 0
    print(
        f"[{row['run_id']}] Starting ALL-FRAME scan of condensed stride-100 trajectory: "
        f"{trajectory}",
        flush=True,
    )
    for frame_index, current_frame in iter_xyz_frames(trajectory):
        frame = current_frame
        frame_count = frame_index + 1
        current_components, _minimum_distance = molecular_components(
            current_frame, oh_cutoff, nh_cutoff, hh_cutoff, bond_scale
        )
        for component in current_components:
            event = flagged_component_event(row, trajectory, frame_index, current_frame, component)
            if event is not None:
                flagged_events.append(event)
        if frame_index == 0 or frame_count % progress_every == 0:
            print(
                f"[{row['run_id']}] ALL-FRAME progress: processed {frame_count} condensed "
                f"frame(s); latest zero-based frame index={frame_index}; "
                f"flagged components so far={len(flagged_events)}",
                flush=True,
            )
    if frame is None:
        raise ValueError(f"No frames found in {trajectory}")
    print(
        f"[{row['run_id']}] Completed ALL-FRAME scan: {frame_count} condensed frame(s), "
        f"zero-based indices 0..{frame_count - 1}, "
        f"{len(flagged_events)} flagged component(s)",
        flush=True,
    )

    elements = Counter(frame.symbols)
    components, minimum_distance = molecular_components(
        frame, oh_cutoff, nh_cutoff, hh_cutoff, bond_scale
    )
    species = Counter(component_formula(frame, component) for component in components)
    largest = max(components, key=len)
    isolated_indices = [component[0] for component in components if len(component) == 1]
    isolated_atoms = [
        {
            "atom_id": index,
            "symbol": frame.symbols[index],
            "position_A": frame.positions[index].tolist(),
        }
        for index in isolated_indices
    ]

    result: dict[str, object] = {
        "run_id": row["run_id"],
        "pressure_GPa": float(row["pressure_GPa"]),
        "temperature_K": float(row["temperature_K"]),
        "ammonia_water_ratio": float(row["ammonia_water_ratio"]),
        "trajectory": str(trajectory),
        "natoms": len(frame.symbols),
        "formula": "".join(f"{name}{count}" for name, count in sorted(elements.items())),
        "molecular_composition": ":".join(
            f"{name}({count})" for name, count in species.most_common()
        ),
        "species": dict(species),
        "elements": dict(sorted(elements.items())),
        "oh_cutoff_A": oh_cutoff,
        "nh_cutoff_A": nh_cutoff,
        "hh_cutoff_A": hh_cutoff,
        "bond_scale": bond_scale,
        "hydrogen_assignment": "nearest_H_O_or_N",
        "molecular_component_count": len(components),
        "largest_component_atoms": len(largest),
        "largest_component_formula": component_formula(frame, largest),
        "isolated_atom_count": len(isolated_atoms),
        "isolated_atom_ids": [atom["atom_id"] for atom in isolated_atoms],
        "isolated_atoms": isolated_atoms,
        "minimum_interatomic_distance_A": minimum_distance,
        "frames_scanned": frame_count,
        "flagged_event_count": len(flagged_events),
        "flagged_events": flagged_events,
    }
    return result


def write_flagged_events_csv(events: list[dict[str, object]], path: Path) -> None:
    fields = [
        "run_id", "pressure_GPa", "temperature_K", "ammonia_water_ratio",
        "condensed_frame_index_0based", "estimated_original_frame_index_0based",
        "condensed_output_stride", "event_type", "species", "component_size",
        "atom_ids_0based", "H_count", "N_count", "O_count", "trajectory",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def write_summary_csv(results: list[dict[str, object]], path: Path) -> None:
    species = sorted({name for result in results for name in result["species"]})
    fields = [
        "run_id", "pressure_GPa", "temperature_K", "ammonia_water_ratio", "natoms",
        "formula", "molecular_composition", "oh_cutoff_A", "nh_cutoff_A", "hh_cutoff_A",
        "bond_scale", "hydrogen_assignment", "molecular_component_count", "largest_component_atoms",
        "largest_component_formula", "isolated_atom_count", "isolated_atom_ids",
        "minimum_interatomic_distance_A", "frames_scanned", "flagged_event_count",
    ] + [f"species_{name}" for name in species]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result[field] for field in fields if field in result}
            row["isolated_atom_ids"] = ";".join(map(str, result["isolated_atom_ids"]))
            row.update({f"species_{name}": result["species"].get(name, 0) for name in species})
            writer.writerow(row)


def plot_result(result: dict[str, object], output_dir: Path, min_count: int) -> Path:
    selected = [(name, count) for name, count in result["species"].items() if count >= min_count]
    selected.sort(key=lambda item: (-item[1], item[0]))
    names = [item[0] for item in selected]
    counts = np.array([item[1] for item in selected], dtype=int)
    figure_width = max(8.0, 0.55 * len(names))
    fig, ax = plt.subplots(figsize=(figure_width, 5.5), constrained_layout=True)
    bars = ax.bar(np.arange(len(names)), counts, color="#3b82a0", edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(np.arange(len(names)), names, rotation=55, ha="right")
    ax.set_ylabel("Number of molecular components")
    ax.set_title(
        f"Final-frame species: {result['run_id']}\n"
        f"P={result['pressure_GPa']:g} GPa, T={result['temperature_K']:g} K, "
        f"NH3/H2O={result['ammonia_water_ratio']:g}; "
        "each H assigned to nearest H/O/N"
    )
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / f"{result['run_id']}_final_species.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if (
        args.workers < 1
        or args.oh_cutoff <= 0
        or args.nh_cutoff <= 0
        or args.hh_cutoff <= 0
        or args.bond_scale <= 0
        or args.min_species_count < 1
        or args.progress_every < 1
    ):
        raise ValueError(
            "--workers, --progress-every, cutoffs, and --min-species-count must be positive"
        )
    rows = selected_rows(args, manifest_rows(args.manifest.resolve()))
    print(
        f"Pressure filter active: running only {TARGET_PRESSURE_GPA:g} GPa conditions; "
        "all other pressures are skipped.",
        flush=True,
    )
    tasks = [
        (
            row,
            str(trajectory_for_run(args.input_dir.resolve(), row["run_id"])),
            args.oh_cutoff,
            args.nh_cutoff,
            args.hh_cutoff,
            args.bond_scale,
            args.progress_every,
        )
        for row in rows
    ]

    results: list[dict[str, object]] = []
    flagged_events: list[dict[str, object]] = []
    # Never create idle processes: the default one-frame/one-condition analysis
    # uses one worker, while multi-condition runs scale up to --workers.
    effective_workers = min(args.workers, len(tasks))
    print(
        f"Using {effective_workers} worker process(es) for {len(tasks)} condition(s). "
        f"Every condensed frame in every selected condition will be scanned; "
        f"progress prints every {args.progress_every} frame(s).",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(analyse_run, task): task[0]["run_id"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            flagged_events.extend(result.pop("flagged_events"))
            results.append(result)
            print(f"Analysed {result['run_id']}: {result['molecular_composition']}")
            print(
                f"  Scanned {result['frames_scanned']} condensed stride-100 frames; "
                f"found {result['flagged_event_count']} flagged component(s)"
            )
            if result["isolated_atom_ids"]:
                print(f"  Isolated atom IDs (0-based): {result['isolated_atom_ids']}")

    order = {row["run_id"]: index for index, row in enumerate(rows)}
    results.sort(key=lambda result: order[result["run_id"]])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        plot_result(result, output_dir, args.min_species_count)
        with (output_dir / f"{result['run_id']}_final_species.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    write_summary_csv(results, output_dir / "final_species_summary.csv")
    flagged_events.sort(
        key=lambda event: (
            order[event["run_id"]],
            event["condensed_frame_index_0based"],
            event["event_type"],
            event["atom_ids_0based"],
        )
    )
    write_flagged_events_csv(flagged_events, output_dir / "flagged_components_all_frames.csv")
    metadata = {
        "frame_indexing": "ZERO-BASED",
        "scan_scope": (
            "ALL condensed frames in EVERY selected condition are scanned. The event CSV is not "
            "limited to the last frame. Only the separate final-species plots summarize last frames."
        ),
        "important_frame_note": (
            "condensed_frame_index_0based is a ZERO-BASED index into the condensed XYZ "
            "outputs, which were written with stride 100. It is not the original trajectory "
            "frame index. estimated_original_frame_index_0based equals "
            "condensed_frame_index_0based * 100."
        ),
        "condensed_output_stride": CONDENSED_OUTPUT_STRIDE,
        "pressure_filter_GPa": TARGET_PRESSURE_GPA,
        "atom_indexing": "ZERO-BASED",
        "event_definitions": {
            "N_or_O_component_without_H": (
                "A connected molecular component containing at least one N or O and no H."
            ),
            "H_only_component": (
                "A connected molecular component made only of H atoms, including H, H2, H3, etc."
            ),
        },
        "hydrogen_assignment": (
            "Each H is assigned to its nearest other H, O, or N using periodic minimum-image distance."
        ),
        "event_csv": "flagged_components_all_frames.csv",
        "event_row_scope": "One row per flagged connected component per condensed frame.",
    }
    with (output_dir / "flagged_components_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote statistics and plots to {output_dir}")


if __name__ == "__main__":
    main()
