from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import os
import re
from collections.abc import Iterator
from pathlib import Path

# Parallel usage:
#   python plot_r09_hot_w_water_autoionization.py --cpu=24
#
# Use process-level parallelism over sampled trajectory frames. Keep BLAS/OpenMP
# threads at 1 so --cpu=24 means 24 worker processes, not nested thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "outputsfull" / "7_25_mdtempramp" 
FS_PER_PS = 1000.0
SPECIES = ("O", "OH", "H2O", "H3O", "H4O_or_more")
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
CANONICAL_O_H_COUNTS = {1, 2, 3}
DEFAULT_H2_CUTOFF = 1.0
DEFAULT_CONNECTIVITY_CUTOFF = 1.5


def find_one(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    if pattern == "*.xyz":
        files = [path for path in files if "checkpoint" not in path.name.lower()] or files
    return files[0] if files else None


def load_thermo(thermo_path: Path) -> dict[str, np.ndarray]:
    if not thermo_path.is_file():
        return {}
    first_line = thermo_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0].strip()
    if not first_line:
        return {}
    header = first_line.lstrip("# ").replace(",", " ").split()
    data = np.atleast_2d(np.genfromtxt(thermo_path, comments="#", dtype=float))
    thermo = {name: data[:, index] for index, name in enumerate(header)}
    if "time_fs" in thermo:
        thermo["time_ps"] = thermo["time_fs"] / FS_PER_PS
    return thermo


def parse_lattice(comment: str) -> np.ndarray:
    match = LATTICE_RE.search(comment)
    if not match:
        raise ValueError(f"Frame comment has no Lattice field: {comment[:120]}")
    values = np.fromstring(match.group(1), sep=" ", dtype=float)
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def iter_xyz_frames(xyz_path: Path) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    with xyz_path.open("r", encoding="utf-8", errors="ignore") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue
            natoms = int(natoms_line)
            comment = handle.readline()
            if not comment:
                raise ValueError(f"{xyz_path}: missing comment line at frame {frame_index}")

            symbols: list[str] = []
            positions = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"{xyz_path}: malformed atom line at frame {frame_index}, atom {atom_index}")
                symbols.append(fields[0])
                positions[atom_index] = [float(fields[1]), float(fields[2]), float(fields[3])]

            yield frame_index, np.array(symbols), positions, parse_lattice(comment)
            frame_index += 1


def iter_sampled_xyz_frames(
    xyz_path: Path,
    stride: int,
    max_frames: int | None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    sampled = 0
    for frame_index, symbols, positions, cell in iter_xyz_frames(xyz_path):
        if frame_index % stride != 0:
            continue
        if max_frames is not None and sampled >= max_frames:
            break
        sampled += 1
        yield frame_index, symbols, positions, cell


def minimum_image_distances(h_positions: np.ndarray, o_positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = h_positions[:, None, :] - o_positions[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def minimum_image_vectors(anchor: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions - anchor
    fractional = delta @ np.linalg.inv(cell)
    return delta - np.round(fractional) @ cell


def minimum_image_vector(from_position: np.ndarray, to_position: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = to_position - from_position
    fractional = delta @ np.linalg.inv(cell)
    return delta - np.round(fractional) @ cell


def water_assignment(
    symbols: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    oh_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oxygen_indices = np.flatnonzero(symbols == "O")
    hydrogen_indices = np.flatnonzero(symbols == "H")
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        raise ValueError("Frame does not contain both O and H atoms.")

    distances = minimum_image_distances(positions[hydrogen_indices], positions[oxygen_indices], cell)
    nearest_o = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(hydrogen_indices.size), nearest_o]
    bonded = nearest_distances <= oh_cutoff
    attached_h = np.bincount(nearest_o[bonded], minlength=oxygen_indices.size).astype(int)
    return oxygen_indices, hydrogen_indices, nearest_o, bonded, attached_h


def oxygen_species_label(n_h: int) -> str:
    if n_h == 0:
        return "O"
    if n_h == 1:
        return "OH"
    if n_h == 2:
        return "H2O"
    if n_h == 3:
        return "H3O"
    return f"H{n_h}O"


def classify_frame(symbols: np.ndarray, positions: np.ndarray, cell: np.ndarray, oh_cutoff: float) -> dict[str, int | float | bool]:
    _oxygen_indices, hydrogen_indices, nearest_o, bonded, attached_h = water_assignment(
        symbols, positions, cell, oh_cutoff
    )
    distances = minimum_image_distances(
        positions[hydrogen_indices],
        positions[np.flatnonzero(symbols == "O")],
        cell,
    )
    nearest_distances = distances[np.arange(hydrogen_indices.size), nearest_o]

    counts = {
        "O": int(np.count_nonzero(attached_h == 0)),
        "OH": int(np.count_nonzero(attached_h == 1)),
        "H2O": int(np.count_nonzero(attached_h == 2)),
        "H3O": int(np.count_nonzero(attached_h == 3)),
        "H4O_or_more": int(np.count_nonzero(attached_h >= 4)),
        "unassigned_H": int(np.count_nonzero(~bonded)),
        "min_nearest_OH_distance_A": float(np.min(nearest_distances)),
        "max_bonded_OH_distance_A": float(np.max(nearest_distances[bonded])) if np.any(bonded) else np.nan,
    }
    counts["autoionized"] = bool(counts["OH"] > 0 and counts["H3O"] > 0)
    counts["non_water_oxygens"] = int(counts["O"] + counts["OH"] + counts["H3O"] + counts["H4O_or_more"])
    return counts


def classify_frame_task(task: tuple[int, np.ndarray, np.ndarray, np.ndarray, float]) -> dict[str, int | float | bool]:
    frame_index, symbols, positions, cell, oh_cutoff = task
    row = classify_frame(symbols, positions, cell, oh_cutoff)
    row["frame"] = frame_index
    return row


def find_h2_candidate(
    symbols: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    oh_cutoff: float,
    h2_cutoff: float,
    connectivity_cutoff: float,
) -> tuple[int, int, float] | None:
    oxygen_indices, hydrogen_indices, nearest_o, bonded, _attached_h = water_assignment(
        symbols, positions, cell, oh_cutoff
    )
    if hydrogen_indices.size < 2:
        return None
    distances = minimum_image_distances(positions[hydrogen_indices], positions[hydrogen_indices], cell)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=None)
    for flat_index in order:
        local_i, local_j = np.unravel_index(int(flat_index), distances.shape)
        if local_i >= local_j:
            continue
        distance = float(distances[local_i, local_j])
        if distance > h2_cutoff:
            break
        same_oxygen = bool(bonded[local_i] and bonded[local_j] and nearest_o[local_i] == nearest_o[local_j])
        if same_oxygen:
            continue
        seed_indices = np.array([hydrogen_indices[local_i], hydrogen_indices[local_j]], dtype=int)
        component = connected_component_indices(positions, cell, seed_indices, connectivity_cutoff)
        if component.size == 2 and np.all(symbols[component] == "H"):
            return int(seed_indices[0]), int(seed_indices[1]), distance
    return None


def connected_component_indices(
    positions: np.ndarray,
    cell: np.ndarray,
    seed_indices: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    distances = pairwise_distances_mic(positions, cell)
    adjacency = distances <= cutoff
    np.fill_diagonal(adjacency, False)
    selected: set[int] = set(int(index) for index in seed_indices)
    frontier = list(selected)
    while frontier:
        current = frontier.pop()
        for neighbor in np.flatnonzero(adjacency[current]):
            neighbor = int(neighbor)
            if neighbor in selected:
                continue
            selected.add(neighbor)
            frontier.append(neighbor)
    return np.array(sorted(selected), dtype=int)


def pairwise_distances_mic(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def unwrap_connected_component(
    positions: np.ndarray,
    cell: np.ndarray,
    component_indices: np.ndarray,
    seed_indices: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    component_set = set(int(index) for index in component_indices)
    primary_seed = int(seed_indices[0])
    unwrapped = {primary_seed: np.zeros(3, dtype=float)}
    frontier = [primary_seed]
    for seed_index in seed_indices[1:]:
        seed_index = int(seed_index)
        if seed_index in component_set and seed_index not in unwrapped:
            unwrapped[seed_index] = minimum_image_vector(positions[primary_seed], positions[seed_index], cell)
            frontier.append(seed_index)
    while frontier:
        current = frontier.pop(0)
        for neighbor in component_indices:
            neighbor = int(neighbor)
            if neighbor in unwrapped or neighbor == current:
                continue
            step = minimum_image_vector(positions[current], positions[neighbor], cell)
            if np.linalg.norm(step) <= cutoff and neighbor in component_set:
                unwrapped[neighbor] = unwrapped[current] + step
                frontier.append(neighbor)
    missing = component_set.difference(unwrapped)
    if missing:
        raise RuntimeError(f"Could not unwrap connected atoms: {sorted(missing)}")
    return np.array([unwrapped[int(index)] for index in component_indices], dtype=float)


def validate_unwrapped_component(
    positions: np.ndarray,
    cell: np.ndarray,
    component_indices: np.ndarray,
    relative_positions: np.ndarray,
    cutoff: float,
) -> None:
    original_distances = pairwise_distances_mic(positions[component_indices], cell)
    unwrapped_distances = np.linalg.norm(
        relative_positions[:, None, :] - relative_positions[None, :, :], axis=2
    )
    mask = original_distances <= cutoff
    if not np.allclose(original_distances[mask], unwrapped_distances[mask], atol=1e-8):
        raise RuntimeError("Unwrapped connected component does not preserve minimum-image neighbor distances.")


def connected_h2_cluster_xyz_block(
    frame_index: int,
    symbols: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    time_ps: float,
    oh_cutoff: float,
    connectivity_cutoff: float,
    seed_indices: np.ndarray,
    vacuum: float,
) -> str:
    component_indices = connected_component_indices(positions, cell, seed_indices, connectivity_cutoff)
    relative_positions = unwrap_connected_component(
        positions, cell, component_indices, seed_indices, connectivity_cutoff
    )
    validate_unwrapped_component(positions, cell, component_indices, relative_positions, connectivity_cutoff)
    cluster_positions = relative_positions + 0.5 * vacuum
    h2_distance = np.linalg.norm(relative_positions[np.where(component_indices == seed_indices[0])[0][0]]
                                 - relative_positions[np.where(component_indices == seed_indices[1])[0][0]])
    lines = [
        f"{len(component_indices)}\n",
        (
            f'cluster_id=1 source_frame={frame_index} source_time_ps={time_ps:.8g} '
            f'h2_seed_atoms="{int(seed_indices[0])},{int(seed_indices[1])}" '
            f'h2_distance_A={h2_distance:.8g} oh_cutoff_A={oh_cutoff:.6g} '
            f'connectivity_cutoff_A={connectivity_cutoff:.6g} saved_atoms=connected_component '
            f'Lattice="{vacuum:.8f} 0 0 0 {vacuum:.8f} 0 0 0 {vacuum:.8f}" '
            'Properties=species:S:1:pos:R:3 pbc="F F F"\n'
        ),
    ]
    lines.extend(
        f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n"
        for symbol, pos in zip(symbols[component_indices], cluster_positions)
    )
    return "".join(lines)


def time_for_frame(thermo: dict[str, np.ndarray], frame_index: int) -> float:
    thermo_time = thermo.get("time_ps")
    return (
        float(thermo_time[frame_index])
        if thermo_time is not None and frame_index < thermo_time.size
        else float(frame_index)
    )


def is_noncanonical_row(row: dict[str, int | float | bool]) -> bool:
    return int(row["O"]) > 0 or int(row["H4O_or_more"]) > 0 or int(row["unassigned_H"]) > 0


def find_first_h2_event(
    xyz_path: Path,
    oh_cutoff: float,
    h2_cutoff: float,
    connectivity_cutoff: float,
    stride: int,
    max_frames: int | None,
) -> tuple[int, tuple[int, int], float]:
    for frame_index, symbols, positions, cell in iter_sampled_xyz_frames(xyz_path, stride, max_frames):
        candidate = find_h2_candidate(symbols, positions, cell, oh_cutoff, h2_cutoff, connectivity_cutoff)
        if candidate is not None:
            h_i, h_j, distance = candidate
            return frame_index, (h_i, h_j), distance
    raise RuntimeError(
        f"No H2 candidate found with H-H cutoff {h2_cutoff:g} A "
        f"and an isolated hydrogen-only connected component at {connectivity_cutoff:g} A."
    )


def write_connected_h2_frame(
    handle,
    frame_index: int,
    symbols: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    time_ps: float,
    first_h2_frame: int,
    seed_indices: np.ndarray,
    oh_cutoff: float,
    connectivity_cutoff: float,
    vacuum: float,
) -> None:
    block = connected_h2_cluster_xyz_block(
        frame_index, symbols, positions, cell, time_ps, oh_cutoff, connectivity_cutoff, seed_indices, vacuum
    )
    block = block.replace("cluster_id=1 ", f"frames_from_first_h2={frame_index - first_h2_frame} ", 1)
    handle.write(block)


def write_first_h2_outputs(
    xyz_path: Path,
    thermo_path: Path,
    output_dir: Path,
    oh_cutoff: float,
    first_frame: int,
    h2_seed_atoms: tuple[int, int],
    connectivity_cutoff: float,
    vacuum: float,
) -> tuple[Path, Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = output_dir / "r09_hot_w_first_h2_connected_cluster.xyz"
    context_path = output_dir / "r09_hot_w_first_h2_connected_context_pm100_frames.xyz"
    thermo = load_thermo(thermo_path)
    start_frame = max(0, first_frame - 100)
    stop_frame = first_frame + 100
    seed_indices = np.array(h2_seed_atoms, dtype=int)
    wrote_cluster = False

    with context_path.open("w", encoding="utf-8") as context_handle:
        for frame_index, symbols, positions, cell in iter_xyz_frames(xyz_path):
            if frame_index > stop_frame:
                break
            if frame_index < start_frame:
                continue
            time_ps = time_for_frame(thermo, frame_index)
            write_connected_h2_frame(
                context_handle,
                frame_index,
                symbols,
                positions,
                cell,
                time_ps,
                first_frame,
                seed_indices,
                oh_cutoff,
                connectivity_cutoff,
                vacuum,
            )
            if frame_index == first_frame:
                block = connected_h2_cluster_xyz_block(
                    frame_index,
                    symbols,
                    positions,
                    cell,
                    time_ps,
                    oh_cutoff,
                    connectivity_cutoff,
                    seed_indices,
                    vacuum,
                )
                cluster_path.write_text(block, encoding="utf-8")
                wrote_cluster = True

    if not wrote_cluster:
        raise RuntimeError(f"First H2 frame {first_frame} was not found in {xyz_path}.")
    return cluster_path, context_path, "H2_connected_component"


def analyze_trajectory(
    xyz_path: Path,
    thermo_path: Path,
    oh_cutoff: float,
    stride: int,
    max_frames: int | None,
    cpu: int,
) -> list[dict[str, int | float | bool]]:
    thermo = load_thermo(thermo_path)
    rows: list[dict[str, int | float | bool]] = []

    tasks = (
        (frame_index, symbols, positions, cell, oh_cutoff)
        for frame_index, symbols, positions, cell in iter_sampled_xyz_frames(xyz_path, stride, max_frames)
    )
    cpu = max(1, cpu)
    if cpu <= 1:
        row_iter = map(classify_frame_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=cpu)
        row_iter = executor.map(classify_frame_task, tasks, chunksize=16)

    try:
        for row in row_iter:
            frame_index = int(row["frame"])
            row["time_ps"] = time_for_frame(thermo, frame_index)
            row["temperature_K"] = (
                float(thermo["temperature_K"][frame_index])
                if "temperature_K" in thermo and frame_index < thermo["temperature_K"].size
                else np.nan
            )
            row["pressure_GPa"] = (
                float(thermo["pressure_GPa"][frame_index])
                if "pressure_GPa" in thermo and frame_index < thermo["pressure_GPa"].size
                else np.nan
            )
            rows.append(row)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not rows:
        raise ValueError("No frames were sampled.")
    return rows


def write_csv(rows: list[dict[str, int | float | bool]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "r09_hot_w_water_autoionization_timeseries.csv"
    columns = [
        "frame",
        "time_ps",
        "temperature_K",
        "pressure_GPa",
        *SPECIES,
        "unassigned_H",
        "non_water_oxygens",
        "autoionized",
        "min_nearest_OH_distance_A",
        "max_bonded_OH_distance_A",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_timeseries(rows: list[dict[str, int | float | bool]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "r09_hot_w_water_autoionization_timeseries.png"
    time = np.array([float(row["time_ps"]) for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True, constrained_layout=True)

    colors = {"H2O": "#2166ac", "OH": "#b2182b", "H3O": "#ef8a62", "O": "#4d4d4d", "H4O_or_more": "#7b3294"}
    for species in ("H2O", "OH", "H3O", "O", "H4O_or_more"):
        values = np.array([int(row[species]) for row in rows])
        if species == "H2O" or np.any(values):
            axes[0].step(time, values, where="post", label=species, linewidth=1.2, color=colors[species])

    axes[1].step(time, [int(row["non_water_oxygens"]) for row in rows], where="post", label="non-H2O oxygens", color="#1b7837")
    axes[1].step(time, [int(row["unassigned_H"]) for row in rows], where="post", label="unassigned H", color="#762a83")
    axes[2].plot(time, [float(row["temperature_K"]) for row in rows], linewidth=1.0, color="#d6604d")
    axes[3].plot(time, [float(row["pressure_GPa"]) for row in rows], linewidth=1.0, color="#4393c3")

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("O-centered species count")
    axes[1].set_ylabel("Anomaly count")
    axes[2].set_ylabel("Temperature (K)")
    axes[3].set_ylabel("Pressure (GPa)")
    axes[3].set_xlabel("Time (ps)")
    axes[0].set_title("r09_hot_w water molecular distribution over time")
    axes[1].set_title("Autoionization candidates: OH and H3O in the same frame")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def summarize_events(rows: list[dict[str, int | float | bool]]) -> str:
    events = [row for row in rows if bool(row["autoionized"])]
    if not events:
        return "No sampled frames contained both OH and H3O with the selected cutoff."
    return (
        f"{len(events)} sampled frame(s) contained both OH and H3O. "
        f"First: frame {events[0]['frame']} at {float(events[0]['time_ps']):.6g} ps. "
        f"Last: frame {events[-1]['frame']} at {float(events[-1]['time_ps']):.6g} ps."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot r09_hot_w water autoionization candidates from an XYZ trajectory.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--xyz", type=Path, default=None)
    parser.add_argument("--thermo", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--oh-cutoff", type=float, default=1.4, help="O-H assignment cutoff in Angstrom.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cpu", "--cores", dest="cpu", type=int, default=1, help="Parallel worker processes for sampled frames.")
    parser.add_argument(
        "--h2-cutoff",
        type=float,
        default=DEFAULT_H2_CUTOFF,
        help="H-H distance cutoff in Angstrom for detecting an H2 candidate.",
    )
    parser.add_argument(
        "--connectivity-cutoff",
        type=float,
        default=DEFAULT_CONNECTIVITY_CUTOFF,
        help="Neighbor graph cutoff in Angstrom for tracking atoms connected to the H2 atoms.",
    )
    parser.add_argument("--cluster-vacuum", type=float, default=18.0, help="Vacuum box size in Angstrom.")
    parser.add_argument(
        "--no-save-first-h2",
        action="store_true",
        help="Disable writing the first H2 connected-component cluster and +/-100-frame context XYZ files.",
    )
    args = parser.parse_args()

    if args.oh_cutoff <= 0:
        parser.error("--oh-cutoff must be positive")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.cpu < 1:
        parser.error("--cpu must be at least 1")
    if args.h2_cutoff <= 0:
        parser.error("--h2-cutoff must be positive")
    if args.connectivity_cutoff <= 0:
        parser.error("--connectivity-cutoff must be positive")
    if args.cluster_vacuum <= 0:
        parser.error("--cluster-vacuum must be positive")

    xyz_path = args.xyz or find_one(args.run_dir, "*.xyz")
    if xyz_path is None:
        raise FileNotFoundError(f"No .xyz trajectory found in {args.run_dir}")
    thermo_path = args.thermo or find_one(args.run_dir, "*thermo*.txt") or Path()
    output_dir = args.output_dir or args.run_dir / "plots"

    print(f"Using {args.cpu} CPU worker process(es)")
    rows = analyze_trajectory(xyz_path, thermo_path, args.oh_cutoff, args.stride, args.max_frames, args.cpu)
    csv_path = write_csv(rows, output_dir)
    plot_path = plot_timeseries(rows, output_dir)
    cluster_path = None
    context_path = None
    h2_frame = None
    h2_seed_atoms = None
    h2_distance = None
    if not args.no_save_first_h2:
        h2_frame, h2_seed_atoms, h2_distance = find_first_h2_event(
            xyz_path,
            args.oh_cutoff,
            args.h2_cutoff,
            args.connectivity_cutoff,
            args.stride,
            args.max_frames,
        )
        cluster_path, context_path, _species_label = write_first_h2_outputs(
            xyz_path,
            thermo_path,
            output_dir,
            args.oh_cutoff,
            h2_frame,
            h2_seed_atoms,
            args.connectivity_cutoff,
            args.cluster_vacuum,
        )

    print(f"Analyzed {len(rows)} sampled frame(s) from {xyz_path}")
    print(f"O-H cutoff: {args.oh_cutoff:g} A")
    print(summarize_events(rows))
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    if h2_frame is not None and h2_seed_atoms is not None and h2_distance is not None:
        thermo = load_thermo(thermo_path)
        h2_time_ps = time_for_frame(thermo, h2_frame)
        print(
            f"First H2 candidate: frame {h2_frame} at {h2_time_ps:.6g} ps, "
            f"H atoms {h2_seed_atoms[0]} and {h2_seed_atoms[1]}, H-H distance {h2_distance:.6g} A"
        )
    if cluster_path is not None and context_path is not None:
        print(f"Saved first H2 connected cluster: {cluster_path}")
        print(f"Saved +/-100-frame connected-component context XYZ: {context_path}")


if __name__ == "__main__":
    main()
