#!/usr/bin/env python3
"""Extract H-H/O-O/N-N closest approaches with fixed O-H/N-H length scales."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_ROOT = SCRIPT_DIR.parents[1]
SHARED_EXTRACTOR = MLIP_ROOT / "codes" / "C_DFTproduction" / "extract_condition_clusters.py"
HELPER_PATH = MLIP_ROOT / "codes" / "A_parityplot" / "7_26_H2pathvalidation" / "extract_training_data.py"
DEFAULT_OUTPUT_DIR = MLIP_ROOT / "outputsfull" / "C3_DFTproductionstopH2_ON"
DATASET_DATE = "2026-09-03"
REQUIRED_PRESSURE_GPA = 100.0
PAIRS = {"H-H": "H", "O-O": "O", "N-N": "N"}
FORMAL_CHARGES = {"H": 1, "N": -3, "O": -2, "S": -2}
CONDITION_RE = re.compile(r"^P(?P<pressure>[0-9p.]+)GPa_R(?P<ratio>[0-9p.]*)$")


def covalent_length(symbol_a: str, symbol_b: str) -> float:
    return float(covalent_radii[atomic_numbers[symbol_a]] + covalent_radii[atomic_numbers[symbol_b]])


OH_COVALENT_LENGTH_A = covalent_length("O", "H")
NH_COVALENT_LENGTH_A = covalent_length("N", "H")
MIN_XH_COVALENT_LENGTH_A = min(OH_COVALENT_LENGTH_A, NH_COVALENT_LENGTH_A)
MAX_XH_COVALENT_LENGTH_A = max(OH_COVALENT_LENGTH_A, NH_COVALENT_LENGTH_A)
CLOSEST_APPROACH_MIN_A = 0.9 * MIN_XH_COVALENT_LENGTH_A
CLOSEST_APPROACH_MAX_A = 1.0 * MIN_XH_COVALENT_LENGTH_A
FORCED_INCLUSION_RADIUS_A = 1.8 * MAX_XH_COVALENT_LENGTH_A
GRAPH_COMPLETION_CUTOFF_A = 1.1 * MAX_XH_COVALENT_LENGTH_A


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared = load_module("c3_shared_extractor", SHARED_EXTRACTOR)
helper = load_module("c3_fragment_helpers", HELPER_PATH)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz-root", type=Path, default=shared.DEFAULT_XYZ_ROOT)
    parser.add_argument("--slurm-dir", type=Path, default=shared.DEFAULT_SLURM_DIR)
    parser.add_argument("--output-xyz", type=Path, default=DEFAULT_OUTPUT_DIR / "condition_production_ON_closest_approach.xyz")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "condition_production_ON_closest_approach_summary.csv")
    parser.add_argument("--stage-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "condition_production_ON_stage_windows.csv")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--max-total-per-pair", type=positive_int, default=1000)
    parser.add_argument("--max-per-condition-per-pair", type=positive_int, default=200)
    parser.add_argument("--max-per-frame-per-pair", type=positive_int, default=10)
    parser.add_argument("--max-abs-charge", type=int, default=None,
                        help="Optional charge filter; disabled by default.")
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--progress-every", type=positive_int, default=50,
                        help="Print progress after this many scanned frames. Default: 50.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def condition_sort_key(condition: str) -> tuple[float, float, str]:
    match = CONDITION_RE.match(condition)
    if match is None:
        return (float("-inf"), float("-inf"), condition)
    return (float(match["pressure"].replace("p", ".")),
            float((match["ratio"] or "0").replace("p", ".")), condition)


def is_required_pressure(condition: str) -> bool:
    match = CONDITION_RE.match(condition)
    return match is not None and float(match["pressure"].replace("p", ".")) == REQUIRED_PRESSURE_GPA


def candidates(frame) -> dict[str, list[tuple[float, int, int]]]:
    distances = helper.pairwise_distances_mic(frame.positions, frame.cell)
    result = {pair: [] for pair in PAIRS}
    for pair, symbol in PAIRS.items():
        indices = np.flatnonzero(frame.symbols == symbol)
        for local_i, atom_i in enumerate(indices[:-1]):
            for atom_j in indices[local_i + 1:]:
                distance = float(distances[int(atom_i), int(atom_j)])
                if CLOSEST_APPROACH_MIN_A <= distance <= CLOSEST_APPROACH_MAX_A:
                    result[pair].append((distance, int(atom_i), int(atom_j)))
        result[pair].sort(key=lambda item: (item[0] - CLOSEST_APPROACH_MIN_A, item[1], item[2]))
    return result


def charge_and_spin(symbols) -> tuple[int, int]:
    charge = sum(FORMAL_CHARGES[str(symbol)] for symbol in symbols)
    electrons = sum(int(atomic_numbers[str(symbol)]) for symbol in symbols) - charge
    return charge, 2 if electrons % 2 else 1


def make_cluster(frame, pair: str, candidate, condition: str, source: Path, fraction: float, args):
    distance, atom_i, atom_j = candidate
    helper_frame = helper.FrameData(frame.condensed_index, frame.symbols, frame.positions, frame.cell)
    selected, positions, radius = helper.select_seed_environment(
        helper_frame, (atom_i, atom_j), FORCED_INCLUSION_RADIUS_A, GRAPH_COMPLETION_CUTOFF_A
    )
    charge, spin = charge_and_spin(frame.symbols[selected])
    if args.max_abs_charge is not None and abs(charge) > args.max_abs_charge:
        raise ValueError(f"abs(charge)={abs(charge)} exceeds {args.max_abs_charge}")
    atoms = Atoms(frame.symbols[selected].tolist(), positions=positions, pbc=False)
    atoms.set_cell([args.vacuum] * 3)
    atoms.positions += 0.5 * args.vacuum
    atoms.info.update({
        "dataset_date": DATASET_DATE, "sample_kind": "closest_approach",
        "candidate_pair": pair, "event": "closest_approach",
        "selection_class": "closest_approach", "condition": condition,
        "source_xyz": str(source), "source_condensed_frame": frame.condensed_index,
        "source_frame_fraction": fraction, "seed_atoms": f"{atom_i},{atom_j}",
        "seed_distance_A": distance, "closest_approach_min_A": CLOSEST_APPROACH_MIN_A,
        "closest_approach_max_A": CLOSEST_APPROACH_MAX_A,
        "oh_covalent_length_A": OH_COVALENT_LENGTH_A, "nh_covalent_length_A": NH_COVALENT_LENGTH_A,
        "forced_inclusion_radius_A": FORCED_INCLUSION_RADIUS_A,
        "graph_completion_cutoff_A": GRAPH_COMPLETION_CUTOFF_A,
        "cluster_build_rule": "all_atoms_within_radius_of_either_seed_then_recursive_graph_completion",
        "environment_radius_A": radius, "charge": charge, "spin": spin,
    })
    return atoms


def main() -> None:
    args = arguments()
    if (args.max_abs_charge is not None and args.max_abs_charge < 0) or args.vacuum <= 0:
        raise ValueError("invalid charge or vacuum limit")
    xyz_files = shared.discover_xyz_files(args.xyz_root)
    xyz_files = {
        condition: path for condition, path in xyz_files.items()
        if is_required_pressure(condition)
    }
    if not xyz_files:
        raise FileNotFoundError(
            f"No P{REQUIRED_PRESSURE_GPA:g}GPa condition trajectories found under {args.xyz_root}"
        )
    if args.condition:
        wrong_pressure = sorted(condition for condition in args.condition if not is_required_pressure(condition))
        if wrong_pressure:
            raise ValueError(
                "This extractor is restricted to P100GPa conditions; rejected: "
                + ", ".join(wrong_pressure)
            )
        missing = set(args.condition) - set(xyz_files)
        if missing:
            raise FileNotFoundError(f"Unknown conditions: {sorted(missing)}")
        xyz_files = {key: value for key, value in xyz_files.items() if key in args.condition}
    conditions = sorted(xyz_files, key=condition_sort_key, reverse=True)
    # All condensed frames are scientifically in scope for this extraction.
    # Slurm stage labels are deliberately not used as an eligibility mask.
    args.stage_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.stage_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "selection_rule"])
        writer.writeheader()
        writer.writerows(
            {"run_id": condition, "selection_rule": "all_condensed_frames_no_stage_mask"}
            for condition in conditions
        )

    accepted: list[Atoms] = []
    rows: list[dict[str, object]] = []
    totals = Counter()
    rejected = Counter()
    for condition in conditions:
        source = xyz_files[condition]
        frames = list(shared.iter_xyz_frames(source))
        n_frames = len(frames)
        frames.reverse()
        per_condition = Counter()
        condition_rejected_start = sum(rejected.values())
        print(
            f"{condition}: scanning {n_frames} frames latest-first with "
            f"{args.workers} worker processes",
            flush=True,
        )
        # Pair-distance matrices are CPU-heavy. Processes provide real
        # multi-core parallelism instead of contending on Python's GIL.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            analyses = executor.map(candidates, frames, chunksize=max(1, len(frames) // (args.workers * 4)))
            for scanned, (frame, found) in enumerate(zip(frames, analyses), start=1):
                fraction = frame.condensed_index / max(n_frames - 1, 1)
                for pair in PAIRS:
                    if totals[pair] >= args.max_total_per_pair or per_condition[pair] >= args.max_per_condition_per_pair:
                        continue
                    for candidate in found[pair][:args.max_per_frame_per_pair]:
                        if totals[pair] >= args.max_total_per_pair or per_condition[pair] >= args.max_per_condition_per_pair:
                            break
                        base = {"condition": condition, "candidate_pair": pair, "source_condensed_frame": frame.condensed_index,
                                "seed_atoms": f"{candidate[1]},{candidate[2]}", "seed_distance_A": candidate[0]}
                        try:
                            cluster = make_cluster(frame, pair, candidate, condition, source, fraction, args)
                        except ValueError as exc:
                            rejected[str(exc)] += 1
                            rows.append({**base, "status": "rejected", "message": str(exc)})
                            continue
                        cluster.info["cluster_id"] = len(accepted) + 1
                        accepted.append(cluster)
                        totals[pair] += 1
                        per_condition[pair] += 1
                        rows.append({**base, "status": "accepted", "message": "", "cluster_id": len(accepted),
                                     "natoms": len(cluster), "charge": cluster.info["charge"], "spin": cluster.info["spin"]})
                if scanned % args.progress_every == 0 or scanned == n_frames:
                    found_counts = ", ".join(
                        f"{pair}={per_condition[pair]}" for pair in PAIRS
                    )
                    print(
                        f"{condition}: scanned {scanned}/{n_frames} frames; "
                        f"accepted this condition: {found_counts}; "
                        f"rejected={sum(rejected.values()) - condition_rejected_start}",
                        flush=True,
                    )
        print(
            f"{condition}: complete; accepted "
            + ", ".join(f"{pair}={per_condition[pair]}" for pair in PAIRS),
            flush=True,
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cluster_id", "status", "message", "candidate_pair", "condition", "source_condensed_frame",
              "seed_atoms", "seed_distance_A", "natoms", "charge", "spin"]
    with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    if not args.dry_run and accepted:
        args.output_xyz.parent.mkdir(parents=True, exist_ok=True)
        write(args.output_xyz, accepted, format="extxyz")
    print(f"accepted={dict(totals)} total={len(accepted)} rejected={sum(rejected.values())}")
    print(f"rejection_reasons={dict(rejected)}")
    print(f"summary={args.summary_csv}")
    if not args.dry_run:
        print(f"xyz={args.output_xyz}")


if __name__ == "__main__":
    main()
