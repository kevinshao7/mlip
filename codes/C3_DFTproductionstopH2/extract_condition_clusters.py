#!/usr/bin/env python3
"""Extract DFT production clusters near H2/N2 formation from all conditions.

Sampling is restricted to the production_hold stage identified from Slurm .err
logs. The first 10% of that production_hold interval is dropped. Per condition,
clusters are built with the current near-pair connected-component logic:
same-element pairs are accepted when formed_cutoff <= distance <= near_cutoff.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_XYZ_ROOT = MLIP_ROOT / "outputsfull" / "B1_conditionsproduction_stride100_xyz"
DEFAULT_SLURM_DIR = MLIP_ROOT / "outputsfull" / "slurm"
DEFAULT_OUTPUT_DIR = MLIP_ROOT / "outputsfull" / "C3_DFTproductionstopH2"
DEFAULT_OUTPUT_XYZ = DEFAULT_OUTPUT_DIR / "condition_production_stopH2_clusters.xyz"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "condition_production_stopH2_clusters_summary.csv"
DEFAULT_STAGE_CSV = DEFAULT_OUTPUT_DIR / "condition_production_stopH2_stage_windows.csv"
DEFAULT_NEAR_GRAPH_CUTOFF_SCALES = {
    "H2": 1.2,
    "O2": 0.8,
    "N2": 0.6,
}
DEFAULT_TOTAL_MD_STEPS = 4_000_000
DEFAULT_MAX_TOTAL_CLUSTERS = 2_000
DEFAULT_MAX_CLUSTERS_PER_CONDITION = 400

PROGRESS_RE = re.compile(r"\|\s*(?P<step>\d+)/(?P<total>\d+)")
STAGE_RE = re.compile(r"stage=(?P<stage>[A-Za-z0-9_]+)")
JOB_ID_RE = re.compile(r"_(\d+)\.err$")
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
CONDITION_RE = re.compile(r"^P(?P<pressure>[0-9p.]+)GPa_R(?P<ratio>[0-9p.]*)$")
DIATOMIC_SYMBOLS = {
    "H2": "H",
    "O2": "O",
    "N2": "N",
}
ACTIVE_MOLECULES = ("N2","H2")
FORMED_COVALENT_SCALES = {
    "H2": 1.5,
    "O2": 1.0,
    "N2": 1.1,
}
NEAR_COVALENT_SCALES = {
    "H2": 1.6,
    "O2": 1.2,
    "N2": 1.3,
}


def covalent_bond_length(symbol: str) -> float:
    return 2.0 * float(covalent_radii[atomic_numbers[symbol]])


MOLECULE_CUTOFFS_A = {
    molecule: FORMED_COVALENT_SCALES[molecule] * covalent_bond_length(symbol)
    for molecule, symbol in DIATOMIC_SYMBOLS.items()
}
NEAR_CUTOFFS_A = {
    molecule: NEAR_COVALENT_SCALES[molecule] * covalent_bond_length(symbol)
    for molecule, symbol in DIATOMIC_SYMBOLS.items()
}

unknown_active_molecules = set(ACTIVE_MOLECULES) - set(DIATOMIC_SYMBOLS)
if unknown_active_molecules:
    raise ValueError(f"Unknown active molecule(s): {sorted(unknown_active_molecules)}")


@dataclass(frozen=True)
class StageWindow:
    run_id: str
    err_path: Path
    production_start_step: int
    production_last_step: int
    usable_start_step: int
    production_start_fraction: float
    production_last_fraction: float
    usable_start_fraction: float
    total_steps_seen: int


@dataclass(frozen=True)
class FrameData:
    condensed_index: int
    symbols: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    comment: str


@dataclass(frozen=True)
class PairCandidate:
    sample_kind: str
    molecule: str
    condition: str
    condensed_frame: int
    frame_fraction: float
    atom_i: int
    atom_j: int
    distance_A: float
    formed_cutoff_A: float
    near_cutoff_A: float


@dataclass(frozen=True)
class FrameResult:
    frame: FrameData
    near_candidates: list[PairCandidate]


def status(message: str) -> None:
    print(message, flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz-root", type=Path, default=DEFAULT_XYZ_ROOT)
    parser.add_argument("--slurm-dir", type=Path, default=DEFAULT_SLURM_DIR)
    parser.add_argument("--output-xyz", type=Path, default=DEFAULT_OUTPUT_XYZ)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--stage-csv", type=Path, default=DEFAULT_STAGE_CSV)
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help="Process only this condition directory name. Repeat to process multiple conditions.",
    )
    parser.add_argument("--max-total-clusters", type=positive_int, default=DEFAULT_MAX_TOTAL_CLUSTERS)
    parser.add_argument(
        "--max-clusters-per-condition",
        type=positive_int,
        default=DEFAULT_MAX_CLUSTERS_PER_CONDITION,
    )
    parser.add_argument("--h2-near-graph-cutoff-scale", type=float, default=DEFAULT_NEAR_GRAPH_CUTOFF_SCALES["H2"])
    parser.add_argument("--o2-near-graph-cutoff-scale", type=float, default=DEFAULT_NEAR_GRAPH_CUTOFF_SCALES["O2"])
    parser.add_argument("--n2-near-graph-cutoff-scale", type=float, default=DEFAULT_NEAR_GRAPH_CUTOFF_SCALES["N2"])
    parser.add_argument("--min-cluster-atoms", type=positive_int, default=5)
    parser.add_argument("--max-cluster-atoms", type=positive_int, default=20)
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--workers", type=positive_int, default=8, help="Parallel worker threads. Default: 8.")
    parser.add_argument(
        "--max-per-frame-per-kind",
        type=positive_int,
        default=100,
        help="Maximum examples to keep from one frame for each output class.",
    )
    parser.add_argument("--progress-every", type=positive_int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def condition_sort_key(condition: str) -> tuple[float, float, str]:
    match = CONDITION_RE.match(condition)
    if not match:
        return (float("-inf"), float("-inf"), condition)
    pressure = float(match.group("pressure").replace("p", "."))
    ratio_text = match.group("ratio") or "0"
    ratio = float(ratio_text.replace("p", "."))
    return (pressure, ratio, condition)


def run_id_from_err(path: Path) -> str | None:
    if not path.name.startswith("cond_") or not path.name.endswith(".err"):
        return None
    stem = path.stem
    match = JOB_ID_RE.search(path.name)
    if not match:
        return None
    suffix = "_" + match.group(1)
    if not stem.endswith(suffix):
        return None
    return stem[len("cond_") : -len(suffix)]


def job_id(path: Path) -> int:
    match = JOB_ID_RE.search(path.name)
    return int(match.group(1)) if match else -1


def parse_stage_window_from_err(run_id: str, path: Path) -> StageWindow | None:
    production_steps: list[int] = []
    total_steps_seen = 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk in re.split(r"[\r\n]+", text):
        if "stage=" not in chunk:
            continue
        progress_matches = list(PROGRESS_RE.finditer(chunk))
        stage_match = STAGE_RE.search(chunk)
        if not progress_matches or stage_match is None:
            continue
        progress = progress_matches[-1]
        step = int(progress.group("step"))
        total_steps_seen = max(total_steps_seen, int(progress.group("total")))
        if stage_match.group("stage") == "production_hold":
            production_steps.append(step)
    if not production_steps:
        return None
    start = min(production_steps)
    last = max(production_steps)
    usable_start = int(math.ceil(start + 0.10 * max(0, last - start)))
    total_steps = total_steps_seen or DEFAULT_TOTAL_MD_STEPS
    return StageWindow(
        run_id=run_id,
        err_path=path,
        production_start_step=start,
        production_last_step=last,
        usable_start_step=usable_start,
        production_start_fraction=start / total_steps,
        production_last_fraction=last / total_steps,
        usable_start_fraction=usable_start / total_steps,
        total_steps_seen=total_steps,
    )


def full_trajectory_window(run_id: str) -> StageWindow:
    return StageWindow(
        run_id=run_id,
        err_path=Path("NO_PRODUCTION_HOLD_FOUND_FULL_TRAJECTORY_FALLBACK"),
        production_start_step=0,
        production_last_step=DEFAULT_TOTAL_MD_STEPS,
        usable_start_step=0,
        production_start_fraction=0.0,
        production_last_fraction=1.0,
        usable_start_fraction=0.0,
        total_steps_seen=DEFAULT_TOTAL_MD_STEPS,
    )


def discover_stage_windows(slurm_dir: Path, run_ids: list[str]) -> tuple[dict[str, StageWindow], list[str]]:
    candidates: dict[str, list[Path]] = {run_id: [] for run_id in run_ids}
    for path in slurm_dir.glob("cond_*.err"):
        run_id = run_id_from_err(path)
        if run_id in candidates:
            candidates[run_id].append(path)

    windows: dict[str, StageWindow] = {}
    missing_production_hold: list[str] = []
    for run_id in run_ids:
        parsed: list[StageWindow] = []
        for path in sorted(candidates[run_id], key=job_id, reverse=True):
            window = parse_stage_window_from_err(run_id, path)
            if window is not None:
                parsed.append(window)
        if not parsed:
            missing_production_hold.append(run_id)
            windows[run_id] = full_trajectory_window(run_id)
            status(f"{run_id}: no production_hold stage found in copied .err logs; scanning full condensed XYZ")
            continue
        windows[run_id] = max(parsed, key=lambda item: (item.production_last_step, job_id(item.err_path)))
        selected = windows[run_id]
        status(
            f"{run_id}: production_hold steps {selected.production_start_step}.."
            f"{selected.production_last_step}; usable starts at {selected.usable_start_step} "
            f"(fractions {selected.production_start_fraction:.6f}.."
            f"{selected.production_last_fraction:.6f}, usable {selected.usable_start_fraction:.6f}) "
            f"from {selected.err_path.name}"
        )
    return windows, missing_production_hold


def discover_xyz_files(xyz_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for run_dir in sorted(path for path in xyz_root.iterdir() if path.is_dir()):
        xyz_files = sorted(run_dir.glob("*.xyz"))
        if len(xyz_files) != 1:
            raise RuntimeError(f"Expected exactly one .xyz under {run_dir}, found {len(xyz_files)}")
        paths[run_dir.name] = xyz_files[0]
    if not paths:
        raise RuntimeError(f"No condition .xyz files found under {xyz_root}")
    return paths


def parse_lattice(comment: str) -> np.ndarray:
    match = LATTICE_RE.search(comment)
    if match is None:
        raise ValueError(f"Frame comment has no Lattice field: {comment[:120]}")
    values = np.fromstring(match.group(1), sep=" ", dtype=float)
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def iter_xyz_frames(xyz_path: Path):
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
                condensed_index=frame_index,
                symbols=np.array(symbols),
                positions=positions,
                cell=parse_lattice(comment),
                comment=comment.rstrip("\n"),
            )
            frame_index += 1


def load_xyz_frames(xyz_path: Path) -> list[FrameData]:
    return list(iter_xyz_frames(xyz_path))


def eligible_frame_indices(xyz_path: Path, window: StageWindow) -> tuple[set[int], int]:
    all_frames: list[int] = []
    condensed_index = 0
    with xyz_path.open("r", encoding="utf-8", errors="ignore") as handle:
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
                raise ValueError(f"{xyz_path}: missing comment at frame {condensed_index}")
            for _ in range(natoms):
                if not handle.readline():
                    raise ValueError(f"{xyz_path}: unexpected EOF at frame {condensed_index}")
            all_frames.append(condensed_index)
            condensed_index += 1

    denominator = max(len(all_frames) - 1, 1)
    eligible = {
        condensed_frame
        for condensed_frame in all_frames
        if window.usable_start_fraction <= condensed_frame / denominator <= window.production_last_fraction
    }
    return eligible, len(all_frames)


def minimum_image_vectors(anchor: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions - anchor
    fractional = delta @ np.linalg.inv(cell)
    return delta - np.round(fractional) @ cell


def pairwise_distances_mic(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def cross_distances_mic(first: np.ndarray, second: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = first[:, None, :] - second[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def candidate_pairs(frame: FrameData, condition: str, frame_fraction: float) -> list[PairCandidate]:
    near: list[PairCandidate] = []

    for molecule in ACTIVE_MOLECULES:
        symbol = DIATOMIC_SYMBOLS[molecule]
        atom_indices = np.flatnonzero(frame.symbols == symbol)
        if atom_indices.size < 2:
            continue
        formed_cutoff = MOLECULE_CUTOFFS_A[molecule]
        near_cutoff = NEAR_CUTOFFS_A[molecule]
        distances = cross_distances_mic(frame.positions[atom_indices], frame.positions[atom_indices], frame.cell)
        for local_i, atom_i in enumerate(atom_indices[:-1]):
            for local_j in range(local_i + 1, atom_indices.size):
                distance = float(distances[local_i, local_j])
                atom_j = int(atom_indices[local_j])
                candidate = PairCandidate(
                    sample_kind="near_formation",
                    molecule=molecule,
                    condition=condition,
                    condensed_frame=frame.condensed_index,
                    frame_fraction=frame_fraction,
                    atom_i=int(atom_i),
                    atom_j=atom_j,
                    distance_A=distance,
                    formed_cutoff_A=float(formed_cutoff),
                    near_cutoff_A=float(near_cutoff),
                )
                if formed_cutoff <= distance <= near_cutoff:
                    near.append(candidate)

    near.sort(key=lambda item: (item.distance_A - item.formed_cutoff_A, item.molecule, item.atom_i, item.atom_j))
    return near


def analyze_frame(payload: tuple[FrameData, str, float]) -> FrameResult:
    frame, condition, frame_fraction = payload
    return FrameResult(frame, candidate_pairs(frame, condition, frame_fraction))


def connected_component_from_pair(
    frame: FrameData,
    seed_atoms: tuple[int, int],
    graph_cutoff_A: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    seed_i, seed_j = seed_atoms
    seed_vector = minimum_image_vectors(frame.positions[seed_i], frame.positions[[seed_j]], frame.cell)[0]
    center = frame.positions[seed_i] + 0.5 * seed_vector

    distances = pairwise_distances_mic(frame.positions, frame.cell)
    selected: set[int] = set()
    unwrapped: dict[int, np.ndarray] = {}

    def add_component(root: int, root_position: np.ndarray) -> None:
        component = {int(root)}
        component_unwrapped = {int(root): root_position.copy()}
        frontier = [int(root)]
        while frontier:
            current = frontier.pop(0)
            for neighbor in np.flatnonzero(distances[current] <= graph_cutoff_A):
                neighbor = int(neighbor)
                if neighbor in component:
                    continue
                component.add(neighbor)
                step = minimum_image_vectors(frame.positions[current], frame.positions[[neighbor]], frame.cell)[0]
                component_unwrapped[neighbor] = component_unwrapped[current] + step
                frontier.append(neighbor)

        for atom_index, position in component_unwrapped.items():
            if atom_index not in unwrapped:
                unwrapped[atom_index] = position
        selected.update(component)

    add_component(int(seed_i), frame.positions[seed_i])
    add_component(int(seed_j), frame.positions[seed_i] + seed_vector)

    selected_indices = np.array(sorted(selected), dtype=int)
    relative_positions = np.array([unwrapped[int(index)] - center for index in selected_indices], dtype=float)
    environment_radius = float(np.max(np.linalg.norm(relative_positions, axis=1))) if selected_indices.size else 0.0
    return selected_indices, relative_positions, environment_radius


def charge_and_spin(symbols: np.ndarray) -> tuple[int, int]:
    charge = int(sum(-2 if symbol == "O" else 1 for symbol in symbols))
    spin = 2 if charge % 2 else 1
    return charge, spin


def make_cluster(
    frame: FrameData,
    candidate: PairCandidate,
    trajectory: Path,
    cluster_id: int,
    graph_cutoff_scales: dict[str, float],
    vacuum: float,
) -> tuple[Atoms, dict[str, object]]:
    graph_cutoff_scale = graph_cutoff_scales[candidate.molecule]
    graph_cutoff_A = graph_cutoff_scale * candidate.distance_A
    selected, relative_positions, actual_radius = connected_component_from_pair(
        frame,
        (candidate.atom_i, candidate.atom_j),
        graph_cutoff_A,
    )
    cluster = Atoms(frame.symbols[selected].tolist(), positions=relative_positions, pbc=False)
    cluster.set_cell([vacuum, vacuum, vacuum])
    cluster.positions += 0.5 * vacuum
    charge, spin = charge_and_spin(frame.symbols[selected])

    metadata: dict[str, object] = {
        "cluster_id": cluster_id,
        "sample_kind": candidate.sample_kind,
        "molecule": candidate.molecule,
        "condition": candidate.condition,
        "covalent_bond_length_A": covalent_bond_length(DIATOMIC_SYMBOLS[candidate.molecule]),
        "formed_covalent_scale": FORMED_COVALENT_SCALES[candidate.molecule],
        "near_covalent_scale": NEAR_COVALENT_SCALES[candidate.molecule],
        "source_xyz": str(trajectory),
        "source_condensed_frame": candidate.condensed_frame,
        "source_frame_fraction": candidate.frame_fraction,
        "seed_atoms": f"{candidate.atom_i},{candidate.atom_j}",
        "seed_distance_A": candidate.distance_A,
        "formed_cutoff_A": candidate.formed_cutoff_A,
        "near_cutoff_A": candidate.near_cutoff_A,
        "distance_margin_to_formed_cutoff_A": candidate.distance_A - candidate.formed_cutoff_A,
        "selection_rule": (
            "same-element pair; near if formed_cutoff<=distance<=near_cutoff; union of connected components "
            "grown separately from both seed atoms using graph edges <= graph_cutoff_scale * seed_distance"
        ),
        "environment_radius_A": actual_radius,
        "graph_cutoff_scale": graph_cutoff_scale,
        "graph_cutoff_A": graph_cutoff_A,
        "charge": charge,
        "spin": spin,
        "natoms": len(cluster),
    }
    for molecule, symbol in DIATOMIC_SYMBOLS.items():
        prefix = molecule.lower()
        metadata[f"{prefix}_covalent_bond_length_A"] = covalent_bond_length(symbol)
        metadata[f"{prefix}_formed_covalent_scale"] = FORMED_COVALENT_SCALES[molecule]
        metadata[f"{prefix}_near_covalent_scale"] = NEAR_COVALENT_SCALES[molecule]
        metadata[f"{prefix}_formed_cutoff_A"] = MOLECULE_CUTOFFS_A[molecule]
        metadata[f"{prefix}_near_cutoff_A"] = NEAR_CUTOFFS_A[molecule]
        metadata[f"{prefix}_near_graph_cutoff_scale"] = graph_cutoff_scales[molecule]
    cluster.info.update(metadata)
    return cluster, metadata


def near_graph_cutoff_scales(args: argparse.Namespace) -> dict[str, float]:
    return {
        "H2": float(args.h2_near_graph_cutoff_scale),
        "O2": float(args.o2_near_graph_cutoff_scale),
        "N2": float(args.n2_near_graph_cutoff_scale),
    }


def keep_candidates(candidates: list[PairCandidate], kept_in_frame: dict[tuple[str, int], int], limit: int) -> list[PairCandidate]:
    kept: list[PairCandidate] = []
    for candidate in candidates:
        key = (candidate.condition, candidate.condensed_frame)
        current = kept_in_frame.get(key, 0)
        if current >= limit:
            continue
        kept.append(candidate)
        kept_in_frame[key] = current + 1
    return kept


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "sample_kind",
        "condition",
        "molecule",
        "source_condensed_frame",
        "source_frame_fraction",
        "seed_atoms",
        "seed_distance_A",
        "covalent_bond_length_A",
        "formed_covalent_scale",
        "near_covalent_scale",
        "formed_cutoff_A",
        "near_cutoff_A",
        "distance_margin_to_formed_cutoff_A",
        "natoms",
        "charge",
        "spin",
        "environment_radius_A",
        "graph_cutoff_scale",
        "graph_cutoff_A",
        "h2_covalent_bond_length_A",
        "h2_formed_covalent_scale",
        "h2_near_covalent_scale",
        "h2_formed_cutoff_A",
        "h2_near_cutoff_A",
        "h2_near_graph_cutoff_scale",
        "o2_covalent_bond_length_A",
        "o2_formed_covalent_scale",
        "o2_near_covalent_scale",
        "o2_formed_cutoff_A",
        "o2_near_cutoff_A",
        "o2_near_graph_cutoff_scale",
        "n2_covalent_bond_length_A",
        "n2_formed_covalent_scale",
        "n2_near_covalent_scale",
        "n2_formed_cutoff_A",
        "n2_near_cutoff_A",
        "n2_near_graph_cutoff_scale",
        "source_xyz",
        "selection_rule",
        "status",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stage_windows(path: Path, windows: dict[str, StageWindow], missing_production_hold: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "err_path",
        "production_start_step",
        "production_last_step",
        "usable_start_step",
        "production_start_fraction",
        "production_last_fraction",
        "usable_start_fraction",
        "selection_rule",
        "frame_mapping_metadata",
        "total_steps_seen",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_id, window in sorted(windows.items(), key=lambda item: condition_sort_key(item[0]), reverse=True):
            is_fallback = run_id in missing_production_hold
            writer.writerow(
                {
                    "run_id": run_id,
                    "err_path": str(window.err_path),
                    "production_start_step": window.production_start_step,
                    "production_last_step": window.production_last_step,
                    "usable_start_step": window.usable_start_step,
                    "production_start_fraction": window.production_start_fraction,
                    "production_last_fraction": window.production_last_fraction,
                    "usable_start_fraction": window.usable_start_fraction,
                    "selection_rule": (
                        "no production_hold .err log found; full condensed trajectory scanned"
                        if is_fallback
                        else (
                            "frame_fraction=condensed_frame/(n_condensed_frames-1); "
                            "keep usable_start_fraction <= frame_fraction <= production_last_fraction"
                        )
                    ),
                    "frame_mapping_metadata": (
                        "No production_hold .err log was available for this condition."
                        if is_fallback
                        else (
                            "No exact condensed-frame-to-MD-step mapping is assumed. "
                            "The .err production_hold window is converted to fractional progress."
                        )
                    ),
                    "total_steps_seen": window.total_steps_seen,
                }
            )


def extract_for_condition(
    condition: str,
    trajectory: Path,
    window: StageWindow,
    args: argparse.Namespace,
    graph_cutoff_scales: dict[str, float],
    cluster_id_start: int,
    target_count: int,
) -> tuple[list[Atoms], list[dict[str, object]]]:
    near_clusters: list[Atoms] = []
    rows: list[dict[str, object]] = []
    kept_in_frame: dict[tuple[str, int], int] = {}
    rejected_too_small = 0
    rejected_too_large = 0
    eligible, n_frames = eligible_frame_indices(trajectory, window)
    if not eligible:
        raise RuntimeError(f"{condition}: no condensed frames in usable production_hold window")
    denominator = max(n_frames - 1, 1)
    status(f"{condition}: loading {len(eligible)} eligible frame(s) from {trajectory.name}")
    frames = load_xyz_frames(trajectory)
    if not frames:
        raise RuntimeError(f"No frames found in {trajectory}")
    frames_desc = [frame for frame in reversed(frames) if frame.condensed_index in eligible]
    status(
        f"{condition}: loaded {len(frames_desc)} eligible frame(s) of {n_frames}; "
        f"analyzing latest production frames first with {args.workers} thread(s)"
    )

    scanned_frames = 0
    batch_size = max(args.workers * 4, 16)

    def consume_result(result: FrameResult) -> None:
        nonlocal rejected_too_small, rejected_too_large
        for candidate in keep_candidates(
            result.near_candidates,
            kept_in_frame,
            args.max_per_frame_per_kind,
        ):
            if len(near_clusters) >= target_count:
                break
            cluster, row = make_cluster(
                result.frame,
                candidate,
                trajectory,
                cluster_id_start + len(near_clusters),
                graph_cutoff_scales,
                args.vacuum,
            )
            if len(cluster) < args.min_cluster_atoms:
                rejected_too_small += 1
                continue
            if len(cluster) > args.max_cluster_atoms:
                rejected_too_large += 1
                continue
            near_clusters.append(cluster)
            rows.append(row)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, len(frames_desc), batch_size):
            batch = frames_desc[batch_start : batch_start + batch_size]
            payloads = [(frame, condition, frame.condensed_index / denominator) for frame in batch]
            for result in executor.map(analyze_frame, payloads):
                scanned_frames += 1
                consume_result(result)
                if scanned_frames % args.progress_every == 0:
                    status(
                        f"{condition}: scanned {scanned_frames} reverse-ordered frames; kept "
                        f"{len(near_clusters)}/{target_count} near-formation examples "
                        f"(rejected <{args.min_cluster_atoms}: {rejected_too_small}, "
                        f">{args.max_cluster_atoms}: {rejected_too_large})"
                    )
                if len(near_clusters) >= target_count:
                    break
            if len(near_clusters) >= target_count:
                break

    status(
        f"{condition}: finished reverse scan after {scanned_frames} frame(s): kept "
        f"{len(near_clusters)} near-formation examples "
        f"(rejected <{args.min_cluster_atoms}: {rejected_too_small}, "
        f">{args.max_cluster_atoms}: {rejected_too_large})"
    )
    if not near_clusters:
        status(f"{condition}: warning no near-formation examples found")
    return near_clusters, rows


def main() -> None:
    args = parse_args()
    graph_cutoff_scales = near_graph_cutoff_scales(args)
    if any(scale <= 0 for scale in graph_cutoff_scales.values()) or args.vacuum <= 0:
        raise ValueError("--*-near-graph-cutoff-scale and --vacuum must be positive")
    if args.min_cluster_atoms > args.max_cluster_atoms:
        raise ValueError("--min-cluster-atoms must be <= --max-cluster-atoms")

    status(
        "Pair cutoffs: "
        + ", ".join(
            f"{name} formed<{MOLECULE_CUTOFFS_A[name]:g} A near={MOLECULE_CUTOFFS_A[name]:g}..{NEAR_CUTOFFS_A[name]:g} A"
            for name in DIATOMIC_SYMBOLS
        )
    )
    status(
        "Stage selection: convert .err production_hold start/end to fractions, "
        "drop the first 10% of that interval, then apply those fractions to condensed XYZ frame indices."
    )

    xyz_files = discover_xyz_files(args.xyz_root)
    if args.condition:
        requested = set(args.condition)
        missing_requested = sorted(requested - set(xyz_files))
        if missing_requested:
            raise FileNotFoundError(
                "Requested condition(s) not found under "
                f"{args.xyz_root}: {', '.join(missing_requested)}"
            )
        xyz_files = {condition: path for condition, path in xyz_files.items() if condition in requested}
        status(f"Condition filter active: {', '.join(sorted(xyz_files))}")
    ordered_conditions = sorted(xyz_files, key=condition_sort_key, reverse=True)
    status("Condition order: " + ", ".join(ordered_conditions))
    windows, missing_production_hold = discover_stage_windows(args.slurm_dir, ordered_conditions)
    write_stage_windows(args.stage_csv, windows, missing_production_hold)
    if missing_production_hold:
        status(
            "Using full-trajectory fallback for conditions with no production_hold in copied Slurm logs: "
            + ", ".join(sorted(missing_production_hold))
        )

    all_clusters: list[Atoms] = []
    all_rows: list[dict[str, object]] = []
    for condition in ordered_conditions:
        trajectory = xyz_files[condition]
        remaining_total = args.max_total_clusters - len(all_clusters)
        if remaining_total <= 0:
            status(f"Reached global cluster cap: {args.max_total_clusters}")
            break
        try:
            condition_target = min(args.max_clusters_per_condition, remaining_total)
            clusters, rows = extract_for_condition(
                condition,
                trajectory,
                windows[condition],
                args,
                graph_cutoff_scales,
                len(all_clusters) + 1,
                condition_target,
            )
        except Exception as exc:
            status(f"{condition}: extraction failed: {exc}")
            all_rows.append(
                {
                    "condition": condition,
                    "sample_kind": "condition",
                    "status": "error",
                    "message": str(exc),
                }
            )
            continue
        all_clusters.extend(clusters)
        all_rows.extend(rows)

    status(f"Total clusters extracted: {len(all_clusters)}")

    if args.dry_run:
        write_summary(args.summary_csv, all_rows)
        status(f"Dry run: summary written to {args.summary_csv}; XYZ was not written")
        return

    if not all_clusters:
        write_summary(args.summary_csv, all_rows)
        status("No accepted near-formation clusters; existing XYZ output was not overwritten")
        status(f"Wrote audit summary:                 {args.summary_csv}")
        return

    args.output_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(args.output_xyz, all_clusters, format="extxyz")
    write_summary(args.summary_csv, all_rows)
    status(f"Wrote near-formation clusters:       {args.output_xyz}")
    status(f"Wrote audit summary:                 {args.summary_csv}")


if __name__ == "__main__":
    main()
