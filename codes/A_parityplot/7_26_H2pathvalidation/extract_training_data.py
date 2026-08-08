#!/usr/bin/env python3
"""Extract isolated-H and unbiased cutoff clusters for DFT fine-tuning.

The output is an extended XYZ geometry set for later ORCA labeling. Clusters are
built from local environments around isolated H atoms or unbiased center atoms.
"""

from __future__ import annotations

import argparse
import csv
import re
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(path for path in SCRIPT_DIR.parents if (path / "outputsfull").is_dir())
DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "outputsfull"
    / "7_25_mdtempramp"
    / "temperature_ramp_seed_525385756_from_pressure_equil_seed_353168294_P_15GPa_T_300K_density_0.2_P_15GPa_300K_to_2340K.xyz"
)
DEFAULT_THERMO = DEFAULT_TRAJECTORY.with_name(DEFAULT_TRAJECTORY.stem + "_thermo.txt")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputsfull" / "7_26_H2pathvalidation"
DEFAULT_OUTPUT_XYZ = DEFAULT_OUTPUT_DIR / "r09_hot_w_h2formation_training_clusters.xyz"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "r09_hot_w_h2formation_training_clusters_summary.csv"
DEFAULT_FIXED_EVENTS = (
    "16677:247:1.7133;"
    "18831:14:1.7261;"
    "19268:262:1.7239;"
    "21730:139:1.7146;"
    "23819:103:1.7027;"
    "24611:293:1.7255;"
    "25776:50:1.7050;"
    "27815:220:1.7206;"
    "29279:158:1.7591;"
    "32560:40:1.7021;"
    "33590:205:1.7003;"
    "34528:52:1.7870;"
    "34578:122:2.1079;"
    "38704:22:1.7273;"
    "39224:229:1.7021;"
    "40975:274:1.7592;"
    "42787:296:1.7827;"
    "44435:266:1.7213"
)

FS_PER_PS = 1000.0
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


@dataclass(frozen=True)
class FrameData:
    index: int
    symbols: np.ndarray
    positions: np.ndarray
    cell: np.ndarray


@dataclass(frozen=True)
class H2Event:
    event_id: int
    frame: int
    seed_atoms: tuple[int, ...]
    h2_distance_A: float


@dataclass(frozen=True)
class ExtractionRequest:
    sample_kind: str
    source_frame: int
    source_time_ps: float
    seed_atoms: tuple[int, ...]
    event_id: int | None = None
    event_frame: int | None = None
    frames_before_event: int | None = None
    frame_offset_from_event: int | None = None
    h2_distance_A: float | None = None


def status(message: str) -> None:
    print(message, flush=True)


def parse_lattice(comment: str) -> np.ndarray:
    match = LATTICE_RE.search(comment)
    if not match:
        raise ValueError(f"Frame comment has no Lattice field: {comment[:120]}")
    values = np.fromstring(match.group(1), sep=" ", dtype=float)
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def iter_xyz_frames(xyz_path: Path) -> Iterator[FrameData]:
    with xyz_path.open("r", encoding="utf-8", errors="ignore") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            stripped = natoms_line.strip()
            if not stripped:
                continue
            natoms = int(stripped)
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

            yield FrameData(
                index=frame_index,
                symbols=np.array(symbols),
                positions=positions,
                cell=parse_lattice(comment),
            )
            frame_index += 1


def load_times_ps(thermo_path: Path) -> np.ndarray:
    if not thermo_path.is_file():
        return np.array([], dtype=float)
    first_line = thermo_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    header = first_line.lstrip("# ").replace(",", " ").split()
    data = np.atleast_2d(np.genfromtxt(thermo_path, comments="#", dtype=float))
    if "time_fs" not in header:
        return np.array([], dtype=float)
    return np.asarray(data[:, header.index("time_fs")] / FS_PER_PS, dtype=float)


def time_for_frame(times_ps: np.ndarray, frame_index: int) -> float:
    if frame_index < times_ps.size:
        return float(times_ps[frame_index])
    return float(frame_index)


def minimum_image_vectors(anchor: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions - anchor
    fractional = delta @ np.linalg.inv(cell)
    return delta - np.round(fractional) @ cell


def pairwise_distances_mic(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def isolated_h_candidates(
    symbols: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    oxygen_exclusion_radius: float,
) -> list[tuple[int, float]]:
    """Return H atoms with no oxygen within the exclusion radius."""
    hydrogen_indices = np.flatnonzero(symbols == "H")
    if hydrogen_indices.size == 0:
        return []

    distances = pairwise_distances_mic(positions, cell)
    np.fill_diagonal(distances, np.inf)
    oxygen_indices = np.flatnonzero(symbols == "O")
    candidates: list[tuple[int, float]] = []

    for seed in hydrogen_indices:
        seed = int(seed)
        nearest_oxygen = (
            float(np.min(distances[seed, oxygen_indices]))
            if oxygen_indices.size
            else float("inf")
        )
        if nearest_oxygen <= oxygen_exclusion_radius:
            continue
        candidates.append((seed, nearest_oxygen))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def is_isolated_h(
    symbols: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    seed_atom: int,
    oxygen_exclusion_radius: float,
) -> tuple[bool, float]:
    seed = int(seed_atom)
    distances = pairwise_distances_mic(positions, cell)
    np.fill_diagonal(distances, np.inf)
    oxygen_indices = np.flatnonzero(symbols == "O")
    nearest_oxygen = (
        float(np.min(distances[seed, oxygen_indices]))
        if oxygen_indices.size
        else float("inf")
    )
    return bool(symbols[seed] == "H" and nearest_oxygen > oxygen_exclusion_radius), nearest_oxygen


def find_h2_events(
    xyz_path: Path,
    h2_cutoff: float,
    oxygen_exclusion_radius: float,
    target_events: int,
    event_stride: int,
    min_event_gap: int,
    start_frame: int,
    max_scan_frames: int | None,
) -> tuple[list[H2Event], int]:
    events: list[H2Event] = []
    del h2_cutoff
    seen_atoms: set[int] = set()
    last_event_frame = -10**12
    total_frames = 0

    for frame in iter_xyz_frames(xyz_path):
        total_frames = frame.index + 1
        if max_scan_frames is not None and frame.index >= max_scan_frames:
            break
        if frame.index < start_frame:
            continue
        if frame.index % event_stride != 0:
            continue
        for seed, nearest_oxygen in isolated_h_candidates(
            frame.symbols, frame.positions, frame.cell, oxygen_exclusion_radius
        ):
            if seed in seen_atoms:
                continue
            if frame.index - last_event_frame < min_event_gap:
                continue
            event = H2Event(
                event_id=len(events) + 1,
                frame=frame.index,
                seed_atoms=(seed,),
                h2_distance_A=nearest_oxygen,
            )
            events.append(event)
            seen_atoms.add(seed)
            last_event_frame = frame.index
            status(
                f"Isolated H event {event.event_id}: frame {event.frame}, "
                f"H atom {seed}, nearest O={nearest_oxygen:.4f} A"
            )
            break
        if len(events) >= target_events:
            break

    if len(events) < target_events and max_scan_frames is None:
        for frame in iter_xyz_frames(xyz_path):
            total_frames = max(total_frames, frame.index + 1)
    return events, total_frames


def parse_fixed_events(spec: str) -> list[H2Event]:
    events: list[H2Event] = []
    for raw_item in spec.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) not in (2, 3):
            raise ValueError(f"Expected fixed event as frame:atom[:nearest_O_A], got {item!r}")
        frame = int(fields[0])
        atom = int(fields[1])
        nearest_oxygen = float(fields[2]) if len(fields) == 3 else float("nan")
        events.append(
            H2Event(
                event_id=len(events) + 1,
                frame=frame,
                seed_atoms=(atom,),
                h2_distance_A=nearest_oxygen,
            )
        )
    if not events:
        raise ValueError("--fixed-events was provided but contained no events")
    return events


def covalent_bond_cutoff(symbol_a: str, symbol_b: str) -> float:
    pair = tuple(sorted((symbol_a, symbol_b)))
    cutoffs = {
        ("H", "H"): 1.05,
        ("H", "O"): 1.35,
        ("H", "N"): 1.35,
        ("H", "S"): 1.65,
    }
    return cutoffs.get(pair, 0.0)


def covalent_fragments(symbols: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> list[np.ndarray]:
    distances = pairwise_distances_mic(positions, cell)
    adjacency = np.zeros(distances.shape, dtype=bool)
    for i, symbol_i in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            cutoff = covalent_bond_cutoff(str(symbol_i), str(symbols[j]))
            if cutoff > 0.0 and distances[i, j] <= cutoff:
                adjacency[i, j] = True
                adjacency[j, i] = True

    visited: set[int] = set()
    fragments: list[np.ndarray] = []
    for seed in range(len(symbols)):
        if seed in visited:
            continue
        selected = {seed}
        frontier = [seed]
        visited.add(seed)
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[current]):
                neighbor = int(neighbor)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                selected.add(neighbor)
                frontier.append(neighbor)
        fragments.append(np.array(sorted(selected), dtype=int))
    return fragments


def unwrap_fragment_positions(
    positions: np.ndarray,
    cell: np.ndarray,
    fragment_indices: np.ndarray,
    anchor_index: int,
) -> dict[int, np.ndarray]:
    fragment_set = set(int(index) for index in fragment_indices)
    distances = pairwise_distances_mic(positions[fragment_indices], cell)
    adjacency = np.zeros(distances.shape, dtype=bool)
    fragment_symbols = None
    # The covalent graph was already used to choose fragments; within a small
    # fragment, connect by the nearest-image distances that define that fragment.
    del fragment_symbols
    for local_i, atom_i in enumerate(fragment_indices):
        for local_j in range(local_i + 1, len(fragment_indices)):
            atom_j = int(fragment_indices[local_j])
            # A generous intrafragment cutoff only affects unwrapping of atoms
            # already assigned to the same covalent fragment.
            if distances[local_i, local_j] <= 1.75:
                adjacency[local_i, local_j] = True
                adjacency[local_j, local_i] = True

    local_for_atom = {int(atom): local for local, atom in enumerate(fragment_indices)}
    unwrapped = {int(anchor_index): positions[int(anchor_index)].copy()}
    frontier = [int(anchor_index)]
    while frontier:
        current = frontier.pop(0)
        current_local = local_for_atom[current]
        for neighbor_local in np.flatnonzero(adjacency[current_local]):
            neighbor = int(fragment_indices[int(neighbor_local)])
            if neighbor not in fragment_set or neighbor in unwrapped:
                continue
            step = minimum_image_vectors(positions[current], positions[[neighbor]], cell)[0]
            unwrapped[neighbor] = unwrapped[current] + step
            frontier.append(neighbor)
    if set(unwrapped) != fragment_set:
        for atom_index in fragment_indices:
            atom_index = int(atom_index)
            if atom_index not in unwrapped:
                unwrapped[atom_index] = positions[int(anchor_index)] + minimum_image_vectors(
                    positions[int(anchor_index)], positions[[atom_index]], cell
                )[0]
    return unwrapped


def unwrap_selected_positions(
    positions: np.ndarray,
    cell: np.ndarray,
    selected_indices: np.ndarray,
    center: np.ndarray,
    completion_radius: float,
) -> dict[int, np.ndarray]:
    selected_set = set(int(index) for index in selected_indices)
    selected_list = [int(index) for index in selected_indices]
    distances = pairwise_distances_mic(positions[selected_indices], cell)
    adjacency = np.zeros(distances.shape, dtype=bool)
    for local_i in range(len(selected_indices)):
        for local_j in range(local_i + 1, len(selected_indices)):
            if distances[local_i, local_j] <= completion_radius:
                adjacency[local_i, local_j] = True
                adjacency[local_j, local_i] = True

    local_for_atom = {int(atom): local for local, atom in enumerate(selected_indices)}
    relative_to_center = minimum_image_vectors(center, positions[selected_indices], cell)
    unwrapped: dict[int, np.ndarray] = {}
    visited: set[int] = set()

    for anchor_local in np.argsort(np.linalg.norm(relative_to_center, axis=1)):
        anchor = selected_list[int(anchor_local)]
        if anchor in visited:
            continue
        unwrapped[anchor] = center + relative_to_center[int(anchor_local)]
        visited.add(anchor)
        frontier = [anchor]
        while frontier:
            current = frontier.pop(0)
            current_local = local_for_atom[current]
            for neighbor_local in np.flatnonzero(adjacency[current_local]):
                neighbor = selected_list[int(neighbor_local)]
                if neighbor not in selected_set or neighbor in visited:
                    continue
                step = minimum_image_vectors(positions[current], positions[[neighbor]], cell)[0]
                unwrapped[neighbor] = unwrapped[current] + step
                visited.add(neighbor)
                frontier.append(neighbor)

    return unwrapped


def select_nearest_fragments(
    frame: FrameData,
    seed_indices: tuple[int, ...],
    center: np.ndarray,
    max_atoms: int,
    min_atoms: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    fragments = covalent_fragments(frame.symbols, frame.positions, frame.cell)
    fragment_for_atom: dict[int, int] = {}
    for fragment_id, fragment in enumerate(fragments):
        for atom_index in fragment:
            fragment_for_atom[int(atom_index)] = fragment_id

    selected_fragment_ids = {fragment_for_atom[int(seed)] for seed in seed_indices}
    selected_count = sum(len(fragments[fragment_id]) for fragment_id in selected_fragment_ids)
    if selected_count > max_atoms:
        raise ValueError(f"Seed molecular fragments contain {selected_count} atoms, above max {max_atoms}.")

    relative_positions = minimum_image_vectors(center, frame.positions, frame.cell)
    fragment_distances: list[tuple[float, int]] = []
    for fragment_id, fragment in enumerate(fragments):
        distance = float(np.min(np.linalg.norm(relative_positions[fragment], axis=1)))
        fragment_distances.append((distance, fragment_id))
    fragment_distances.sort()

    for _distance, fragment_id in fragment_distances:
        if selected_count >= min_atoms:
            break
        if fragment_id in selected_fragment_ids:
            continue
        candidate_count = selected_count + len(fragments[fragment_id])
        if candidate_count > max_atoms:
            continue
        selected_fragment_ids.add(fragment_id)
        selected_count = candidate_count

    if selected_count < min_atoms:
        raise ValueError(f"Nearest complete fragments only reached {selected_count} atoms, need {min_atoms}.")

    selected_indices = np.concatenate([fragments[fragment_id] for fragment_id in sorted(selected_fragment_ids)])
    selected_indices = np.array(sorted(int(index) for index in selected_indices), dtype=int)
    final_positions_by_atom: dict[int, np.ndarray] = {}
    for fragment_id in selected_fragment_ids:
        fragment = fragments[fragment_id]
        anchor = int(fragment[np.argmin(np.linalg.norm(relative_positions[fragment], axis=1))])
        final_positions_by_atom.update(unwrap_fragment_positions(frame.positions, frame.cell, fragment, anchor))
    final_positions = np.array(
        [minimum_image_vectors(center, np.array([final_positions_by_atom[int(index)]]), frame.cell)[0] for index in selected_indices],
        dtype=float,
    )
    radius = float(np.max(np.linalg.norm(final_positions, axis=1))) if selected_indices.size else 0.0
    return selected_indices, final_positions, radius


def select_seed_environment(
    frame: FrameData,
    seed_indices: tuple[int, ...],
    environment_radius: float,
    completion_radius: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    seeds = tuple(int(seed) for seed in seed_indices)
    center = frame.positions[seeds[0]]
    distances_to_seeds = np.vstack(
        [np.linalg.norm(minimum_image_vectors(frame.positions[seed], frame.positions, frame.cell), axis=1) for seed in seeds]
    )
    selected = set(int(index) for index in np.flatnonzero(np.min(distances_to_seeds, axis=0) <= environment_radius))
    selected.update(seeds)

    distances = pairwise_distances_mic(frame.positions, frame.cell)
    frontier = list(selected)
    while frontier:
        current = frontier.pop(0)
        for neighbor in np.flatnonzero(distances[current] <= completion_radius):
            neighbor = int(neighbor)
            if neighbor in selected:
                continue
            selected.add(neighbor)
            frontier.append(neighbor)

    selected_indices = np.array(sorted(selected), dtype=int)
    unwrapped = unwrap_selected_positions(
        frame.positions,
        frame.cell,
        selected_indices,
        center,
        completion_radius,
    )
    relative_positions = np.array([unwrapped[int(index)] - center for index in selected_indices], dtype=float)
    radius = float(np.max(np.linalg.norm(relative_positions, axis=1))) if selected_indices.size else 0.0
    return selected_indices, relative_positions, radius


def cluster_from_request(
    frame: FrameData,
    request: ExtractionRequest,
    max_atoms: int,
    min_atoms: int,
    vacuum: float,
    charge: int,
    spin_multiplicity: int,
    source_xyz: Path,
    h2_cutoff: float,
    oxygen_exclusion_radius: float,
    h2_environment_radius: float,
    h2_completion_radius: float,
) -> tuple[Atoms, float]:
    current_isolated_h = False
    seed_distance = np.nan
    cluster_build_rule = "nearest_complete_covalent_fragments"
    if request.event_id is not None:
        seed = int(request.seed_atoms[0])
        current_isolated_h, seed_distance = is_isolated_h(
            frame.symbols,
            frame.positions,
            frame.cell,
            seed,
            oxygen_exclusion_radius,
        )
        selected, relative_positions, radius = select_seed_environment(
            frame,
            request.seed_atoms,
            h2_environment_radius,
            h2_completion_radius,
        )
        cluster_build_rule = "isolated_H_4A_environment_with_1p5A_recursive_completion"
    else:
        center = frame.positions[int(request.seed_atoms[0])]
        selected, relative_positions, radius = select_nearest_fragments(
            frame,
            request.seed_atoms,
            center,
            max_atoms,
            min_atoms,
        )

    cluster = Atoms(frame.symbols[selected].tolist(), positions=relative_positions, pbc=False)
    cluster.set_cell([vacuum, vacuum, vacuum])
    cluster.positions += 0.5 * vacuum
    cluster.info.update(
        {
            "sample_kind": request.sample_kind,
            "source_xyz": str(source_xyz),
            "source_frame": int(request.source_frame),
            "source_time_ps": float(request.source_time_ps),
            "seed_atoms": ",".join(str(index) for index in request.seed_atoms),
            "seed_distance_A": float(seed_distance),
            "seed_nearest_oxygen_distance_A": float(seed_distance) if request.event_id is not None else np.nan,
            "isolated_h_current_frame": bool(current_isolated_h),
            "environment_radius_A": float(radius),
            "cluster_build_rule": cluster_build_rule,
            "h2_environment_radius_A": float(h2_environment_radius) if request.event_id is not None else np.nan,
            "h2_completion_radius_A": float(h2_completion_radius) if request.event_id is not None else np.nan,
            "oxygen_exclusion_radius_A": float(oxygen_exclusion_radius) if request.event_id is not None else np.nan,
            "max_atoms_rule": "not_applied_to_isolated_H_radius_clusters" if request.event_id is not None else "reject_not_truncate_fragments",
            "charge": int(charge),
            "spin": int(spin_multiplicity),
        }
    )
    if request.event_id is not None:
        cluster.info["event_id"] = int(request.event_id)
        cluster.info["event_frame"] = int(request.event_frame)
        cluster.info["frames_before_event"] = int(request.frames_before_event)
        cluster.info["frame_offset_from_event"] = int(request.frame_offset_from_event)
        cluster.info["h2_definition"] = "isolated_H_with_no_O_within_oxygen_exclusion_radius"
    if request.h2_distance_A is not None:
        cluster.info["event_nearest_oxygen_distance_A"] = float(request.h2_distance_A)
    return cluster, radius


def cluster_from_request_job(
    frame: FrameData,
    request: ExtractionRequest,
    max_atoms: int,
    min_atoms: int,
    vacuum: float,
    charge: int,
    spin_multiplicity: int,
    source_xyz: Path,
    h2_cutoff: float,
    oxygen_exclusion_radius: float,
    h2_environment_radius: float,
    h2_completion_radius: float,
) -> tuple[int, ExtractionRequest, Atoms | None, float | None, str | None]:
    try:
        cluster, environment_radius = cluster_from_request(
            frame,
            request,
            max_atoms,
            min_atoms,
            vacuum,
            charge,
            spin_multiplicity,
            source_xyz,
            h2_cutoff,
            oxygen_exclusion_radius,
            h2_environment_radius,
            h2_completion_radius,
        )
    except ValueError as exc:
        return int(request.event_id), request, None, None, str(exc)
    return int(request.event_id), request, cluster, environment_radius, None


def unbiased_cluster_from_frame(
    frame: FrameData,
    request: ExtractionRequest,
    rng: random.Random,
    max_atoms: int,
    min_atoms: int,
    vacuum: float,
    charge: int,
    spin_multiplicity: int,
    source_xyz: Path,
    attempts: int,
    h2_cutoff: float,
    oxygen_exclusion_radius: float,
) -> tuple[Atoms, float, ExtractionRequest]:
    tried: set[int] = set()
    for _attempt in range(max(1, attempts)):
        if len(tried) >= len(frame.symbols):
            break
        seed = rng.randrange(len(frame.symbols))
        while seed in tried and len(tried) < len(frame.symbols):
            seed = rng.randrange(len(frame.symbols))
        tried.add(seed)
        candidate = ExtractionRequest(
            sample_kind=request.sample_kind,
            source_frame=request.source_frame,
            source_time_ps=request.source_time_ps,
            seed_atoms=(seed,),
        )
        try:
            cluster, environment_radius = cluster_from_request(
                frame,
                candidate,
                max_atoms,
                min_atoms,
                vacuum,
                charge,
                spin_multiplicity,
                source_xyz,
                h2_cutoff,
                oxygen_exclusion_radius,
                np.nan,
                np.nan,
            )
        except ValueError:
            continue
        return cluster, environment_radius, candidate
    raise ValueError(
        f"Could not extract an unbiased cluster from frame {request.source_frame} "
        f"after {max(1, attempts)} seed attempts."
    )


def formation_requests(
    events: list[H2Event],
    frames_before_event: int,
    frames_after_event: int,
    times_ps: np.ndarray,
    min_frame: int,
    max_frame: int,
) -> list[ExtractionRequest]:
    requests: list[ExtractionRequest] = []
    for event in events:
        start = event.frame - frames_before_event
        stop = event.frame + frames_after_event
        if start < min_frame:
            raise ValueError(
                f"Isolated-H event {event.event_id} at frame {event.frame} does not have "
                f"{frames_before_event} frames before it at or after sampling start frame {min_frame}."
            )
        if stop > max_frame:
            raise ValueError(
                f"Isolated-H event {event.event_id} at frame {event.frame} does not have "
                f"{frames_after_event} frames after it within available frames ending at {max_frame}."
            )
        window_frames = list(range(start, event.frame + 1)) + list(range(event.frame + 1, stop + 1))
        for frame_index in window_frames:
            requests.append(
                ExtractionRequest(
                    sample_kind="isolated_h",
                    source_frame=frame_index,
                    source_time_ps=time_for_frame(times_ps, frame_index),
                    seed_atoms=event.seed_atoms,
                    event_id=event.event_id,
                    event_frame=event.frame,
                    frames_before_event=event.frame - frame_index,
                    frame_offset_from_event=frame_index - event.frame,
                    h2_distance_A=event.h2_distance_A,
                )
            )
    return requests


def extract_h2_window_clusters(
    xyz_path: Path,
    events: list[H2Event],
    frames_before_event: int,
    frames_after_event: int,
    times_ps: np.ndarray,
    min_frame: int,
    max_frame: int,
    max_atoms: int,
    min_atoms: int,
    vacuum: float,
    charge: int,
    spin_multiplicity: int,
    target_events: int,
    h2_cutoff: float,
    oxygen_exclusion_radius: float,
    h2_environment_radius: float,
    h2_completion_radius: float,
    workers: int,
) -> tuple[list[tuple[ExtractionRequest, Atoms, float]], list[H2Event], set[int]]:
    frames_per_event = frames_before_event + 1 + frames_after_event
    requests_by_frame: dict[int, list[ExtractionRequest]] = {}
    failures: dict[int, list[str]] = {}
    event_requests: dict[int, list[ExtractionRequest]] = {}
    event_results: dict[int, list[tuple[ExtractionRequest, Atoms, float]]] = {}

    for event in events:
        try:
            requests = formation_requests([event], frames_before_event, frames_after_event, times_ps, min_frame, max_frame)
        except ValueError as exc:
            failures.setdefault(event.event_id, []).append(str(exc))
            continue
        event_requests[event.event_id] = requests
        for request in requests:
            requests_by_frame.setdefault(request.source_frame, []).append(request)

    if workers == 1:
        for frame in iter_xyz_frames(xyz_path):
            frame_requests = requests_by_frame.get(frame.index)
            if not frame_requests:
                continue
            for request in frame_requests:
                event_id, request, cluster, environment_radius, error = cluster_from_request_job(
                    frame,
                    request,
                    max_atoms,
                    min_atoms,
                    vacuum,
                    charge,
                    spin_multiplicity,
                    xyz_path,
                    h2_cutoff,
                    oxygen_exclusion_radius,
                    h2_environment_radius,
                    h2_completion_radius,
                )
                if error is not None:
                    failures.setdefault(event_id, []).append(error)
                    continue
                event_results.setdefault(event_id, []).append((request, cluster, float(environment_radius)))
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for frame in iter_xyz_frames(xyz_path):
                frame_requests = requests_by_frame.get(frame.index)
                if not frame_requests:
                    continue
                for request in frame_requests:
                    futures.append(
                        executor.submit(
                            cluster_from_request_job,
                            frame,
                            request,
                            max_atoms,
                            min_atoms,
                            vacuum,
                            charge,
                            spin_multiplicity,
                            xyz_path,
                            h2_cutoff,
                            oxygen_exclusion_radius,
                            h2_environment_radius,
                            h2_completion_radius,
                        )
                    )
            for future in as_completed(futures):
                event_id, request, cluster, environment_radius, error = future.result()
                if error is not None:
                    failures.setdefault(event_id, []).append(error)
                    continue
                event_results.setdefault(event_id, []).append((request, cluster, float(environment_radius)))

    selected_results: list[tuple[ExtractionRequest, Atoms, float]] = []
    selected_events: list[H2Event] = []
    selected_frames: set[int] = set()
    event_by_id = {event.event_id: event for event in events}
    for event in events:
        results = event_results.get(event.event_id, [])
        if failures.get(event.event_id) or len(results) != frames_per_event:
            reason = "; ".join(failures.get(event.event_id, ["incomplete event window"]))
            status(f"Skipping isolated-H event {event.event_id} at frame {event.frame}: {reason}")
            continue
        order = {request.source_frame: i for i, request in enumerate(event_requests[event.event_id])}
        results.sort(key=lambda item: order[item[0].source_frame])
        selected_results.extend(results)
        selected_events.append(event_by_id[event.event_id])
        selected_frames.update(request.source_frame for request, _cluster, _cutoff in results)
        if len(selected_events) >= target_events:
            break

    return selected_results, selected_events, selected_frames


def extract_unbiased_clusters(
    xyz_path: Path,
    count: int,
    excluded_frames: set[int],
    start_frame: int,
    rng: random.Random,
    max_atoms: int,
    min_atoms: int,
    vacuum: float,
    charge: int,
    spin_multiplicity: int,
    seed_attempts: int,
    times_ps: np.ndarray,
    h2_cutoff: float,
    oxygen_exclusion_radius: float,
) -> list[tuple[ExtractionRequest, Atoms, float]]:
    if count == 0:
        return []

    results: list[tuple[ExtractionRequest, Atoms, float]] = []
    skipped = 0
    for frame in iter_xyz_frames(xyz_path):
        if frame.index < start_frame or frame.index in excluded_frames:
            continue
        request = ExtractionRequest(
            sample_kind="unbiased",
            source_frame=frame.index,
            source_time_ps=time_for_frame(times_ps, frame.index),
            seed_atoms=(-1,),
        )
        try:
            cluster, environment_radius, request = unbiased_cluster_from_frame(
                frame,
                request,
                rng,
                max_atoms,
                min_atoms,
                vacuum,
                charge,
                spin_multiplicity,
                xyz_path,
                seed_attempts,
                h2_cutoff,
                oxygen_exclusion_radius,
            )
        except ValueError:
            skipped += 1
            continue
        results.append((request, cluster, environment_radius))
        if len(results) >= count:
            break
    if len(results) < count:
        raise RuntimeError(
            f"Only extracted {len(results)} unbiased clusters after skipping {skipped} invalid frames; need {count}."
        )
    if skipped:
        status(f"Skipped {skipped} unbiased frames that could not produce {min_atoms}-{max_atoms} atom clusters")
    return results


def count_xyz_frames(xyz_path: Path) -> int:
    total = 0
    for frame in iter_xyz_frames(xyz_path):
        total = frame.index + 1
    return total


def self_test_pbc_backfolding() -> None:
    frame = FrameData(
        index=0,
        symbols=np.array(["H", "O", "H"]),
        positions=np.array(
            [
                [9.8, 5.0, 5.0],
                [0.4, 5.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=float,
        ),
        cell=np.eye(3) * 10.0,
    )
    selected, relative_positions, radius = select_seed_environment(
        frame,
        (0,),
        environment_radius=3.0,
        completion_radius=1.6,
    )
    if selected.tolist() != [0, 1]:
        raise AssertionError(f"Expected boundary-crossing H/O pair to be selected, got {selected.tolist()}")
    if not np.isclose(np.linalg.norm(relative_positions[1] - relative_positions[0]), 0.6):
        raise AssertionError(f"Expected unwrapped H/O distance 0.6 A, got {relative_positions.tolist()}")
    cluster, _radius = cluster_from_request(
        frame,
        ExtractionRequest(sample_kind="isolated_h", source_frame=0, source_time_ps=0.0, seed_atoms=(0,), event_id=1, event_frame=0, frames_before_event=0, frame_offset_from_event=0),
        max_atoms=20,
        min_atoms=1,
        vacuum=24.0,
        charge=0,
        spin_multiplicity=1,
        source_xyz=Path("synthetic.xyz"),
        h2_cutoff=1.2,
        oxygen_exclusion_radius=0.5,
        h2_environment_radius=3.0,
        h2_completion_radius=1.6,
    )
    if not np.isclose(cluster.get_distance(0, 1, mic=False), 0.6):
        raise AssertionError("Backfolded cluster coordinates did not preserve the boundary-crossing H/O distance")
    status(f"PBC self-test passed: selected={selected.tolist()}, radius={radius:.3f} A")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--thermo", type=Path, default=DEFAULT_THERMO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_XYZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--events", type=int, default=20, help="Maximum number of isolated-H events to extract.")
    parser.add_argument(
        "--event-candidates",
        type=int,
        default=20,
        help="Maximum candidate isolated-H events to scan before selecting valid windows.",
    )
    parser.add_argument(
        "--fixed-events",
        type=str,
        default=DEFAULT_FIXED_EVENTS,
        help="Use fixed isolated-H events instead of scanning; format: frame:atom[:nearest_O_A];...",
    )
    parser.add_argument(
        "--frames-before-event",
        type=int,
        default=8,
        help="Frames to extract before each isolated-H event frame.",
    )
    parser.add_argument(
        "--frames-after-event",
        type=int,
        default=1,
        help="Frames to extract after each isolated-H event frame.",
    )
    parser.add_argument("--unbiased-clusters", type=int, default=0)
    parser.add_argument(
        "--h2-cutoff",
        type=float,
        default=1.2,
        help="Unused for isolated-H detection; kept for compatibility with older commands.",
    )
    parser.add_argument(
        "--oxygen-exclusion-radius",
        type=float,
        default=1.7,
        help="Select H atoms only if they have no O within this Angstrom radius.",
    )
    parser.add_argument("--max-atoms", type=int, default=20)
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=10,
        help="Reject small components; invalid unbiased clusters are resampled.",
    )
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument(
        "--h2-environment-radius",
        type=float,
        default=3.0,
        help="For isolated-H windows, include all atoms within this Angstrom radius of the event H atom.",
    )
    parser.add_argument(
        "--h2-completion-radius",
        type=float,
        default=1.5,
        help="For isolated-H windows, recursively include atoms within this Angstrom distance of selected atoms.",
    )
    parser.add_argument("--event-stride", type=int, default=1)
    parser.add_argument("--min-event-gap", type=int, default=50)
    parser.add_argument("--max-scan-frames", type=int, default=None)
    parser.add_argument("--unbiased-min-frame", type=int, default=0)
    parser.add_argument(
        "--sample-start-fraction",
        type=float,
        default=0.3,
        help="Only sample frames at or after this fraction of the trajectory length.",
    )
    parser.add_argument("--unbiased-seed-attempts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="CPU worker processes for isolated-H cluster construction.")
    parser.add_argument("--self-test-pbc", action="store_true", help="Run a synthetic PBC/backfolding check and exit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--charge", type=int, default=0, help="ORCA finite-cluster charge metadata; default neutral.")
    parser.add_argument("--spin", type=int, default=1, help="ORCA spin multiplicity metadata; default singlet.")
    args = parser.parse_args()

    if args.self_test_pbc:
        self_test_pbc_backfolding()
        return

    if args.events <= 0:
        parser.error("--events must be positive")
    if args.event_candidates <= 0:
        parser.error("--event-candidates must be positive")
    if args.frames_before_event < 0:
        parser.error("--frames-before-event cannot be negative")
    if args.frames_after_event < 0:
        parser.error("--frames-after-event cannot be negative")
    h2_frames_per_event = args.frames_before_event + 1 + args.frames_after_event
    if args.unbiased_clusters < 0:
        parser.error("--unbiased-clusters cannot be negative")
    if args.max_atoms < args.min_atoms:
        parser.error("--max-atoms must be >= --min-atoms")
    if args.event_stride <= 0:
        parser.error("--event-stride must be positive")
    if args.unbiased_seed_attempts <= 0:
        parser.error("--unbiased-seed-attempts must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not 0.0 <= args.sample_start_fraction < 1.0:
        parser.error("--sample-start-fraction must be in [0, 1)")
    if args.h2_cutoff <= 0:
        parser.error("--h2-cutoff must be positive")
    if args.oxygen_exclusion_radius <= 0:
        parser.error("--oxygen-exclusion-radius must be positive")
    if args.h2_environment_radius <= 0:
        parser.error("--h2-environment-radius must be positive")
    if args.h2_completion_radius <= 0:
        parser.error("--h2-completion-radius must be positive")
    if not args.trajectory.is_file():
        parser.error(f"--trajectory does not exist: {args.trajectory}")
    if not args.thermo.is_file():
        parser.error(f"--thermo does not exist: {args.thermo}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    status(f"Reading frame times from {args.thermo}")
    times_ps = load_times_ps(args.thermo)
    total_frame_estimate = int(times_ps.size)
    if total_frame_estimate == 0:
        status(f"Counting frames in {args.trajectory}")
        total_frame_estimate = count_xyz_frames(args.trajectory)
    start_frame = int(np.floor(args.sample_start_fraction * total_frame_estimate))
    start_frame = max(start_frame, int(args.unbiased_min_frame))
    if args.fixed_events:
        candidate_events = parse_fixed_events(args.fixed_events)[: args.events]
        total_frames = total_frame_estimate
        status(f"Using {len(candidate_events)} fixed isolated-H events")
    else:
        event_search_start_frame = start_frame + args.frames_before_event
        event_scan_limit = min(args.events, args.event_candidates)

        status(
            f"Scanning isolated-H events in {args.trajectory} from frame {event_search_start_frame} "
            f"({args.sample_start_fraction:.3g} of trajectory), requiring "
            f"{args.frames_before_event} frames before and {args.frames_after_event} frames after each event, "
            f"up to {event_scan_limit} events"
        )
        candidate_events, total_frames = find_h2_events(
            args.trajectory,
            args.h2_cutoff,
            args.oxygen_exclusion_radius,
            event_scan_limit,
            args.event_stride,
            args.min_event_gap,
            event_search_start_frame,
            args.max_scan_frames,
        )
        total_frames = max(total_frames, total_frame_estimate)
    status(f"Trajectory frames available for sampling: {total_frames}")
    last_frame = total_frames - 1

    rng = random.Random(args.seed)
    status(f"Validating candidate isolated-H windows until up to {args.events} complete events are selected")
    status(f"Using {args.workers} worker processes for isolated-H cluster construction")
    h2_results, selected_events, excluded = extract_h2_window_clusters(
        args.trajectory,
        candidate_events,
        args.frames_before_event,
        args.frames_after_event,
        times_ps,
        start_frame,
        last_frame,
        args.max_atoms,
        args.min_atoms,
        args.vacuum,
        args.charge,
        args.spin,
        args.events,
        args.h2_cutoff,
        args.oxygen_exclusion_radius,
        args.h2_environment_radius,
        args.h2_completion_radius,
        args.workers,
    )
    if selected_events:
        status(
            f"Selected {len(selected_events)} complete isolated-H events: "
            + ", ".join(str(event.frame) for event in selected_events)
        )
    else:
        status("Selected 0 complete isolated-H events")
    status(f"Extracting {args.unbiased_clusters} unbiased clusters after isolated-H windows")
    unbiased_results = extract_unbiased_clusters(
        args.trajectory,
        args.unbiased_clusters,
        excluded,
        start_frame,
        rng,
        args.max_atoms,
        args.min_atoms,
        args.vacuum,
        args.charge,
        args.spin,
        args.unbiased_seed_attempts,
        times_ps,
        args.h2_cutoff,
        args.oxygen_exclusion_radius,
    )

    clusters: list[Atoms] = []
    summary_rows: list[dict[str, object]] = []
    ordered_results = h2_results + unbiased_results
    if not ordered_results:
        raise RuntimeError("No clusters selected: no complete isolated-H event frames or unbiased clusters were extracted.")
    status(f"Writing {len(h2_results)} isolated-H clusters first, then {len(unbiased_results)} unbiased clusters")
    for request, cluster, environment_radius in ordered_results:
        cluster.info["cluster_id"] = len(clusters) + 1
        symbols = cluster.get_chemical_symbols()
        summary_rows.append(
            {
                "cluster_id": len(clusters) + 1,
                "sample_kind": request.sample_kind,
                "source_frame": request.source_frame,
                "source_time_ps": f"{request.source_time_ps:.8g}",
                "event_id": "" if request.event_id is None else request.event_id,
                "event_frame": "" if request.event_frame is None else request.event_frame,
                "frames_before_event": "" if request.frames_before_event is None else request.frames_before_event,
                "frame_offset_from_event": "" if request.frame_offset_from_event is None else request.frame_offset_from_event,
                "seed_atoms": ",".join(str(index) for index in request.seed_atoms),
                "seed_distance_A": f"{cluster.info.get('seed_distance_A', np.nan):.8g}",
                "seed_nearest_oxygen_distance_A": f"{cluster.info.get('seed_nearest_oxygen_distance_A', np.nan):.8g}",
                "isolated_h_current_frame": cluster.info.get("isolated_h_current_frame", False),
                "natoms": len(cluster),
                "formula": cluster.get_chemical_formula(),
                "n_H": symbols.count("H"),
                "n_O": symbols.count("O"),
                "n_N": symbols.count("N"),
                "n_S": symbols.count("S"),
                "environment_radius_A": f"{environment_radius:.6g}",
                "cluster_build_rule": cluster.info.get("cluster_build_rule", ""),
                "h2_environment_radius_A": f"{cluster.info.get('h2_environment_radius_A', np.nan):.8g}",
                "h2_completion_radius_A": f"{cluster.info.get('h2_completion_radius_A', np.nan):.8g}",
                "oxygen_exclusion_radius_A": f"{cluster.info.get('oxygen_exclusion_radius_A', np.nan):.8g}",
                "h2_definition": "" if request.event_id is None else "isolated_H_with_no_O_within_oxygen_exclusion_radius",
                "charge": args.charge,
                "spin": args.spin,
            }
        )
        clusters.append(cluster)

    expected = len(selected_events) * h2_frames_per_event + args.unbiased_clusters
    if len(clusters) != expected:
        raise RuntimeError(f"Extracted {len(clusters)} clusters, expected {expected}.")

    write(args.output, clusters, format="extxyz")
    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    sizes = np.array([len(cluster) for cluster in clusters], dtype=int)
    status(f"Saved {len(clusters)} clusters to {args.output}")
    status(f"Summary: {args.summary}")
    status(f"Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}")
    status(f"Charge/spin metadata for ORCA finite clusters: charge={args.charge}, spin={args.spin}")


if __name__ == "__main__":
    main()
