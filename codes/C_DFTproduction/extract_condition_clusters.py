#!/usr/bin/env python3
"""Extract DFT production clusters from condensed condition-production XYZ files.

Sampling is restricted to the production_hold stage identified from Slurm .err
logs. The first 10% of that production_hold interval is dropped. Per condition,
the target is 100 isolated-H local-environment clusters and 100 random
complete-fragment clusters.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from ase import Atoms
from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_XYZ_ROOT = MLIP_ROOT / "outputsfull" / "B1_conditionsproduction_stride100_xyz"
DEFAULT_SLURM_DIR = MLIP_ROOT / "outputsfull" / "slurm"
DEFAULT_OUTPUT_DIR = MLIP_ROOT / "outputsfull" / "C_DFTproduction"
DEFAULT_OUTPUT_XYZ = DEFAULT_OUTPUT_DIR / "condition_production_dft_clusters.xyz"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "condition_production_dft_clusters_summary.csv"
DEFAULT_STAGE_CSV = DEFAULT_OUTPUT_DIR / "condition_production_stage_windows.csv"
HELPER_PATH = MLIP_ROOT / "codes" / "A_parityplot" / "7_26_H2pathvalidation" / "extract_training_data.py"

DEFAULT_TOTAL_MD_STEPS = 4_000_000
DEFAULT_PRESSURE_RAMP_STEPS = 400_000
DEFAULT_TEMPERATURE_RAMP_STEPS = 400_000
DEFAULT_CLUSTERS_PER_CONDITION = 200
DEFAULT_ISOLATED_PER_CONDITION = 100
DEFAULT_RANDOM_PER_CONDITION = 100

PROGRESS_RE = re.compile(r"\|\s*(?P<step>\d+)/(?P<total>\d+)")
STAGE_RE = re.compile(r"stage=(?P<stage>[A-Za-z0-9_]+)")
JOB_ID_RE = re.compile(r"_(\d+)\.err$")
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
WORKER_HELPER: ModuleType | None = None


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
class SampleRequest:
    sample_kind: str
    condition: str
    condensed_frame: int
    frame_fraction: float
    seed_atom: int
    nearest_oxygen_A: float | None = None


@dataclass(frozen=True)
class BuildRequest:
    order_index: int
    condition: str
    xyz_path: Path
    frame: FrameData
    sample_kind: str
    seed_atom: int
    nearest_oxygen_A: float | None
    frame_fraction: float
    random_seed: int


def status(message: str) -> None:
    print(message, flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 4000 condition-production clusters from condensed XYZ trajectories."
    )
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
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--clusters-per-condition", type=positive_int, default=DEFAULT_CLUSTERS_PER_CONDITION)
    parser.add_argument("--isolated-per-condition", type=positive_int, default=DEFAULT_ISOLATED_PER_CONDITION)
    parser.add_argument("--random-per-condition", type=positive_int, default=DEFAULT_RANDOM_PER_CONDITION)
    parser.add_argument("--oxygen-exclusion-radius", type=float, default=1.7)
    parser.add_argument("--isolated-environment-radius", type=float, default=4.5)
    parser.add_argument("--completion-radius", type=float, default=1.5)
    parser.add_argument("--min-cluster-atoms", type=positive_int, default=20)
    parser.add_argument("--random-min-atoms", type=positive_int, default=25)
    parser.add_argument("--random-max-atoms", type=positive_int, default=60)
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=8,
        help="Number of CPU processes used to build clusters within each condition. Default: 8.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h2_extract_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def helper_for_worker() -> ModuleType:
    global WORKER_HELPER
    if WORKER_HELPER is None:
        WORKER_HELPER = load_helper()
    return WORKER_HELPER


def init_cluster_worker() -> None:
    helper_for_worker()


def parse_lattice(comment: str) -> np.ndarray:
    match = LATTICE_RE.search(comment)
    if not match:
        raise ValueError(f"Frame comment has no Lattice field: {comment[:120]}")
    values = np.fromstring(match.group(1), sep=" ", dtype=float)
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def iter_xyz_frames(xyz_path: Path):
    with xyz_path.open("r", encoding="utf-8", errors="ignore") as handle:
        condensed_index = 0
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
                raise ValueError(f"{xyz_path}: missing comment line at frame {condensed_index}")
            symbols: list[str] = []
            positions = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"{xyz_path}: malformed atom line at frame {condensed_index}, atom {atom_index}")
                symbols.append(fields[0])
                positions[atom_index] = [float(fields[1]), float(fields[2]), float(fields[3])]
            yield FrameData(
                condensed_index=condensed_index,
                symbols=np.array(symbols),
                positions=positions,
                cell=parse_lattice(comment),
                comment=comment.rstrip("\n"),
            )
            condensed_index += 1


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
            status(f"{run_id}: no production_hold stage found in copied .err logs")
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


def eligible_frame_indices(xyz_path: Path, window: StageWindow) -> tuple[list[int], int]:
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

    if not all_frames:
        return [], 0

    denominator = max(len(all_frames) - 1, 1)
    frames: list[int] = []
    for condensed_frame in all_frames:
        frame_fraction = condensed_frame / denominator
        if window.usable_start_fraction <= frame_fraction <= window.production_last_fraction:
            frames.append(condensed_frame)
    return frames, len(all_frames)


def frame_to_helper(frame: FrameData, helper: ModuleType):
    return helper.FrameData(
        index=frame.condensed_index,
        symbols=frame.symbols,
        positions=frame.positions,
        cell=frame.cell,
    )


def charge_and_spin(symbols: np.ndarray) -> tuple[int, int]:
    charge = int(sum(-2 if symbol == "O" else 1 for symbol in symbols))
    spin = 2 if charge % 2 else 1
    return charge, spin


def make_isolated_cluster(
    frame: FrameData,
    frame_fraction: float,
    seed: int,
    nearest_oxygen: float,
    condition: str,
    xyz_path: Path,
    helper: ModuleType,
    args: argparse.Namespace,
) -> Atoms:
    selected, relative_positions, radius = helper.select_seed_environment(
        frame_to_helper(frame, helper),
        (seed,),
        args.isolated_environment_radius,
        args.completion_radius,
    )
    build_rule = (
        f"isolated_H_{args.isolated_environment_radius:g}A_environment_"
        f"with_{args.completion_radius:g}A_recursive_completion"
    )
    cluster = Atoms(frame.symbols[selected].tolist(), positions=relative_positions, pbc=False)
    cluster.set_cell([args.vacuum, args.vacuum, args.vacuum])
    cluster.positions += 0.5 * args.vacuum
    charge, spin = charge_and_spin(frame.symbols[selected])
    cluster.info.update(
        {
            "sample_kind": "isolated_h",
            "condition": condition,
            "source_xyz": str(xyz_path),
            "source_condensed_frame": int(frame.condensed_index),
            "source_frame_fraction": float(frame_fraction),
            "seed_atoms": str(seed),
            "seed_nearest_oxygen_distance_A": float(nearest_oxygen),
            "environment_radius_A": float(radius),
            "cluster_build_rule": build_rule,
            "oxygen_exclusion_radius_A": float(args.oxygen_exclusion_radius),
            "isolated_environment_radius_A": float(args.isolated_environment_radius),
            "completion_radius_A": float(args.completion_radius),
            "charge": int(charge),
            "spin": int(spin),
        }
    )
    return cluster


def make_random_cluster(
    frame: FrameData,
    frame_fraction: float,
    seed: int,
    condition: str,
    xyz_path: Path,
    helper: ModuleType,
    args: argparse.Namespace,
) -> Atoms:
    selected, relative_positions, radius = helper.select_nearest_fragments(
        frame_to_helper(frame, helper),
        (seed,),
        frame.positions[seed],
        args.random_max_atoms,
        args.random_min_atoms,
    )
    cluster = Atoms(frame.symbols[selected].tolist(), positions=relative_positions, pbc=False)
    cluster.set_cell([args.vacuum, args.vacuum, args.vacuum])
    cluster.positions += 0.5 * args.vacuum
    charge, spin = charge_and_spin(frame.symbols[selected])
    cluster.info.update(
        {
            "sample_kind": "random",
            "condition": condition,
            "source_xyz": str(xyz_path),
            "source_condensed_frame": int(frame.condensed_index),
            "source_frame_fraction": float(frame_fraction),
            "seed_atoms": str(seed),
            "environment_radius_A": float(radius),
            "cluster_build_rule": "nearest_complete_covalent_fragments",
            "max_atoms_rule": "reject_not_truncate_fragments",
            "charge": int(charge),
            "spin": int(spin),
        }
    )
    return cluster


def build_cluster_job(payload: tuple[BuildRequest, argparse.Namespace]) -> tuple[Atoms | None, dict[str, object]]:
    request, args = payload
    helper = helper_for_worker()
    rng = random.Random(request.random_seed)
    seed_atom = request.seed_atom

    try:
        if request.sample_kind == "isolated_h":
            cluster = make_isolated_cluster(
                request.frame,
                request.frame_fraction,
                request.seed_atom,
                float(request.nearest_oxygen_A),
                request.condition,
                request.xyz_path,
                helper,
                args,
            )
        else:
            cluster = None
            tried: set[int] = set()
            for _attempt in range(50):
                candidate_seed = rng.randrange(len(request.frame.symbols))
                if candidate_seed in tried:
                    continue
                tried.add(candidate_seed)
                try:
                    cluster = make_random_cluster(
                        request.frame,
                        request.frame_fraction,
                        candidate_seed,
                        request.condition,
                        request.xyz_path,
                        helper,
                        args,
                    )
                    seed_atom = candidate_seed
                    break
                except ValueError:
                    continue
            if cluster is None:
                raise ValueError("could not build random complete-fragment cluster after 50 seeds")
    except ValueError as exc:
        return None, {
            "condition": request.condition,
            "sample_kind": request.sample_kind,
            "sample_order": request.order_index,
            "status": "error",
            "message": str(exc),
            "source_condensed_frame": request.frame.condensed_index,
            "source_frame_fraction": request.frame_fraction,
            "seed_atom": seed_atom,
        }

    if len(cluster) < args.min_cluster_atoms:
        return None, {
            "condition": request.condition,
            "sample_kind": request.sample_kind,
            "sample_order": request.order_index,
            "status": "rejected",
            "message": f"cluster has {len(cluster)} atoms, below minimum {args.min_cluster_atoms}; resampling",
            "source_condensed_frame": request.frame.condensed_index,
            "source_frame_fraction": request.frame_fraction,
            "seed_atom": seed_atom,
            "n_atoms": len(cluster),
            "charge": cluster.info["charge"],
            "spin": cluster.info["spin"],
            "environment_radius_A": cluster.info["environment_radius_A"],
        }

    cluster.info["sample_order"] = int(request.order_index)
    return cluster, {
        "condition": request.condition,
        "sample_kind": request.sample_kind,
        "sample_order": request.order_index,
        "status": "ok",
        "message": "",
        "source_condensed_frame": request.frame.condensed_index,
        "source_frame_fraction": request.frame_fraction,
        "seed_atom": seed_atom,
        "n_atoms": len(cluster),
        "charge": cluster.info["charge"],
        "spin": cluster.info["spin"],
        "environment_radius_A": cluster.info["environment_radius_A"],
    }


def extract_for_condition(
    condition: str,
    xyz_path: Path,
    window: StageWindow,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[Atoms], list[dict[str, object]]]:
    helper = load_helper()
    status(f"{condition}: scanning eligible frames in {xyz_path.name}")
    eligible_list, n_condensed_frames = eligible_frame_indices(xyz_path, window)
    eligible = set(eligible_list)
    if not eligible:
        raise RuntimeError(f"{condition}: no condensed frames in usable production_hold window")
    frame_fraction_denominator = max(n_condensed_frames - 1, 1)
    status(
        f"{condition}: {len(eligible)} of {n_condensed_frames} condensed frames after ramp/drop filtering; "
        "selection uses frame_fraction=condensed_frame/(n_condensed_frames-1)"
    )

    isolated_candidates: list[tuple[int, int, float]] = []
    random_frame_indices: list[int] = []
    eligible_frames: dict[int, FrameData] = {}
    scanned_eligible = 0
    for frame in iter_xyz_frames(xyz_path):
        if frame.condensed_index not in eligible:
            continue
        eligible_frames[frame.condensed_index] = frame
        scanned_eligible += 1
        if scanned_eligible % 250 == 0:
            status(
                f"{condition}: scanned {scanned_eligible}/{len(eligible)} eligible frames; "
                f"isolated-H candidates so far {len(isolated_candidates)}"
            )
        random_frame_indices.append(frame.condensed_index)
        candidates = helper.isolated_h_candidates(
            frame.symbols,
            frame.positions,
            frame.cell,
            args.oxygen_exclusion_radius,
        )
        for seed, nearest_oxygen in candidates:
            isolated_candidates.append((frame.condensed_index, int(seed), float(nearest_oxygen)))

    rng.shuffle(isolated_candidates)
    rng.shuffle(random_frame_indices)
    status(
        f"{condition}: candidate pool has {len(isolated_candidates)} isolated-H candidates "
        f"and {len(random_frame_indices)} random frames; targeting "
        f"{args.isolated_per_condition} accepted isolated-H and "
        f"{args.random_per_condition} accepted random clusters with at least "
        f"{args.min_cluster_atoms} atoms"
    )

    accepted_clusters: list[Atoms] = []
    accepted_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    request_counter = 0
    size_stats: dict[str, list[int]] = {"isolated_h": [], "random": []}

    def run_build_batch(build_requests: list[BuildRequest]) -> list[tuple[Atoms | None, dict[str, object]]]:
        if not build_requests:
            return []
        status(
            f"{condition}: building {len(build_requests)} {build_requests[0].sample_kind} candidate(s) "
            f"with {args.workers} worker process(es)"
        )
        ordered_results: list[tuple[Atoms | None, dict[str, object]] | None] = [None] * len(build_requests)
        local_index = {request.order_index: index for index, request in enumerate(build_requests)}
        if args.workers == 1:
            for request in build_requests:
                ordered_results[local_index[request.order_index]] = build_cluster_job((request, args))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_cluster_worker) as executor:
                futures = {
                    executor.submit(build_cluster_job, (request, args)): request.order_index
                    for request in build_requests
                }
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    order_index = futures[future]
                    ordered_results[local_index[order_index]] = future.result()
                    if completed % 25 == 0:
                        status(f"{condition}: completed {completed}/{len(build_requests)} candidate builds")
        return [result for result in ordered_results if result is not None]

    def accept_from_pool(
        sample_kind: str,
        target_count: int,
        candidates: list[tuple[int, int, float | None]],
        output_order_start: int,
    ) -> tuple[list[Atoms], list[dict[str, object]], list[dict[str, object]], int]:
        nonlocal request_counter
        accepted_kind_clusters: list[Atoms] = []
        accepted_kind_rows: list[dict[str, object]] = []
        audit_kind_rows: list[dict[str, object]] = []
        cursor = 0
        batch_size = max(args.workers * 4, 32)
        while len(accepted_kind_clusters) < target_count and cursor < len(candidates):
            batch_candidates = candidates[cursor : cursor + batch_size]
            cursor += len(batch_candidates)
            build_requests: list[BuildRequest] = []
            for frame_index, seed, nearest_oxygen in batch_candidates:
                frame = eligible_frames[frame_index]
                request_counter += 1
                build_requests.append(
                    BuildRequest(
                        order_index=request_counter,
                        condition=condition,
                        xyz_path=xyz_path,
                        frame=frame,
                        sample_kind=sample_kind,
                        seed_atom=seed,
                        nearest_oxygen_A=nearest_oxygen,
                        frame_fraction=frame.condensed_index / frame_fraction_denominator,
                        random_seed=rng.randrange(2**63),
                    )
                )
            for cluster, row in run_build_batch(build_requests):
                audit_kind_rows.append(row)
                n_atoms = row.get("n_atoms")
                if n_atoms not in (None, ""):
                    size_stats[sample_kind].append(int(n_atoms))
                if cluster is None:
                    continue
                if len(accepted_kind_clusters) >= target_count:
                    continue
                final_order = output_order_start + len(accepted_kind_clusters)
                cluster.info["sample_order"] = int(final_order)
                row["sample_order"] = final_order
                accepted_kind_clusters.append(cluster)
                accepted_kind_rows.append(row)
                if len(accepted_kind_clusters) % 25 == 0:
                    status(f"{condition}: accepted {len(accepted_kind_clusters)}/{target_count} {sample_kind} clusters")
        if len(accepted_kind_clusters) < target_count:
            status(
                f"{condition}: warning only accepted {len(accepted_kind_clusters)}/{target_count} "
                f"{sample_kind} clusters after exhausting {len(candidates)} candidate(s)"
            )
        rejected = sum(1 for row in audit_kind_rows if row.get("status") == "rejected")
        errors = sum(1 for row in audit_kind_rows if row.get("status") == "error")
        status(
            f"{condition}: {sample_kind} accepted {len(accepted_kind_clusters)}/{target_count}; "
            f"rejected small {rejected}; errors {errors}; candidates consumed {min(cursor, len(candidates))}"
        )
        return accepted_kind_clusters, accepted_kind_rows, audit_kind_rows, cursor

    isolated_pool = [(frame_index, seed, nearest_oxygen) for frame_index, seed, nearest_oxygen in isolated_candidates]
    random_pool = [(frame_index, -1, None) for frame_index in random_frame_indices]
    isolated_clusters, isolated_rows, isolated_audit, _isolated_used = accept_from_pool(
        "isolated_h",
        args.isolated_per_condition,
        isolated_pool,
        0,
    )
    random_clusters, random_rows, random_audit, _random_used = accept_from_pool(
        "random",
        args.random_per_condition,
        random_pool,
        args.isolated_per_condition,
    )

    accepted_clusters.extend(isolated_clusters)
    accepted_clusters.extend(random_clusters)
    accepted_rows.extend(isolated_rows)
    accepted_rows.extend(random_rows)
    audit_rows.extend(isolated_audit)
    audit_rows.extend(random_audit)

    for sample_kind in ("isolated_h", "random"):
        sizes = size_stats[sample_kind]
        if sizes:
            status(
                f"{condition}: {sample_kind} candidate cluster sizes min={min(sizes)}, "
                f"max={max(sizes)}; accepted min="
                f"{min(len(cluster) for cluster in accepted_clusters if cluster.info.get('sample_kind') == sample_kind) if any(cluster.info.get('sample_kind') == sample_kind for cluster in accepted_clusters) else 'NA'}, "
                f"accepted max="
                f"{max(len(cluster) for cluster in accepted_clusters if cluster.info.get('sample_kind') == sample_kind) if any(cluster.info.get('sample_kind') == sample_kind for cluster in accepted_clusters) else 'NA'}"
            )

    rows = accepted_rows + [
        row for row in audit_rows
        if row.get("status") != "ok" and row not in accepted_rows
    ]
    status(
        f"{condition}: isolated {len(isolated_clusters)}/{args.isolated_per_condition}, "
        f"random {len(random_clusters)}/{args.random_per_condition}, eligible frames {len(eligible)}"
    )
    return accepted_clusters, rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
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
        for run_id, window in sorted(windows.items()):
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
                        "frame_fraction=condensed_frame/(n_condensed_frames-1); "
                        "keep usable_start_fraction <= frame_fraction <= production_last_fraction"
                    ),
                    "frame_mapping_metadata": (
                        "No exact condensed-frame-to-MD-step mapping is assumed. "
                        "The .err production_hold window is converted to fractional progress, "
                        "and condensed XYZ frame i is compared as i/(n_condensed_frames-1)."
                    ),
                    "total_steps_seen": window.total_steps_seen,
                }
            )
        for run_id in sorted(missing_production_hold):
            writer.writerow(
                {
                    "run_id": run_id,
                    "err_path": "NO_PRODUCTION_HOLD_FOUND_IN_ERR",
                    "production_start_step": "",
                    "production_last_step": "",
                    "usable_start_step": "",
                    "production_start_fraction": "",
                    "production_last_fraction": "",
                    "usable_start_fraction": "",
                    "selection_rule": (
                        "frame_fraction=condensed_frame/(n_condensed_frames-1); "
                        "keep usable_start_fraction <= frame_fraction <= production_last_fraction"
                    ),
                    "frame_mapping_metadata": (
                        "No exact condensed-frame-to-MD-step mapping is assumed. "
                        "The .err production_hold window is converted to fractional progress, "
                        "and condensed XYZ frame i is compared as i/(n_condensed_frames-1)."
                    ),
                    "total_steps_seen": "",
                }
            )


def main() -> None:
    args = parse_args()
    if args.isolated_per_condition + args.random_per_condition != args.clusters_per_condition:
        raise ValueError("--isolated-per-condition + --random-per-condition must equal --clusters-per-condition")
    status(
        "Stage selection: convert .err production_hold start/end to fractions of logged total steps, "
        "drop the first 10% of that production_hold interval in fraction space, then apply those "
        "fractions to condensed XYZ frame indices."
    )
    status(
        "Frame metadata: no exact condensed-frame-to-MD-step mapping is assumed; "
        "clusters record source_condensed_frame and source_frame_fraction."
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
    windows, missing_production_hold = discover_stage_windows(args.slurm_dir, sorted(xyz_files))
    write_stage_windows(args.stage_csv, windows, missing_production_hold)
    if missing_production_hold:
        status(
            "Skipping conditions with no production_hold in copied Slurm logs: "
            + ", ".join(sorted(missing_production_hold))
        )

    all_clusters: list[Atoms] = []
    all_rows: list[dict[str, object]] = []
    rng = random.Random(args.seed)
    active_conditions: list[tuple[str, Path, StageWindow, int]] = []
    for condition, xyz_path in sorted(xyz_files.items()):
        if condition not in windows:
            all_rows.append(
                {
                    "condition": condition,
                    "sample_kind": "condition",
                    "status": "skipped",
                    "message": "no production_hold stage found in Slurm .err logs",
                }
            )
            continue
        active_conditions.append((condition, xyz_path, windows[condition], rng.randrange(2**63)))

    status(
        f"Processing {len(active_conditions)} condition(s). Per condition target: "
        f"{args.isolated_per_condition} isolated-H clusters + "
        f"{args.random_per_condition} unbiased random clusters = "
        f"{args.clusters_per_condition} clusters. Cluster builds use {args.workers} worker process(es)."
    )
    for condition, xyz_path, window, condition_seed in active_conditions:
        try:
            clusters, rows = extract_for_condition(
                condition,
                xyz_path,
                window,
                args,
                random.Random(condition_seed),
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
        status(f"{condition}: finished with {sum(1 for row in rows if row.get('status') == 'ok')} ok rows")
        all_clusters.extend(clusters)
        all_rows.extend(rows)

    ok_rows = [row for row in all_rows if row.get("status") == "ok"]
    status(f"Total clusters extracted: {len(ok_rows)}")
    if all_clusters:
        sizes = [len(cluster) for cluster in all_clusters]
        status(f"Accepted cluster sizes: min={min(sizes)}, max={max(sizes)}")
        for sample_kind in ("isolated_h", "random"):
            kind_sizes = [len(cluster) for cluster in all_clusters if cluster.info.get("sample_kind") == sample_kind]
            if kind_sizes:
                status(f"Accepted {sample_kind} sizes: min={min(kind_sizes)}, max={max(kind_sizes)}")
    rejected_small = [row for row in all_rows if row.get("status") == "rejected"]
    if rejected_small:
        status(f"Rejected undersized clusters and resampled: {len(rejected_small)}")
    target_total = len(active_conditions) * args.clusters_per_condition
    if len(ok_rows) != target_total:
        status(
            "Warning: extracted cluster count differs from target "
            f"{target_total}"
        )

    if args.dry_run:
        write_summary(args.summary_csv, all_rows)
        status(f"Dry run: summary written to {args.summary_csv}; no XYZ written")
        return

    args.output_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(args.output_xyz, all_clusters, format="extxyz")
    write_summary(args.summary_csv, all_rows)
    status(f"Wrote clusters: {args.output_xyz}")
    status(f"Wrote summary:  {args.summary_csv}")
    status(f"Wrote stages:   {args.stage_csv}")


if __name__ == "__main__":
    main()
