from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import sys

import numpy as np
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes" / "7_7b_clustervalidation"))
import extract_clusters as base


DEFAULT_INTERACTION_CUTOFF = 1.7
DEFAULT_BOND_FCT = 1.0
DEFAULT_TARGET_ATOMS = 12
DEFAULT_MIN_ATOMS = 10
DEFAULT_MAX_ATOMS = 14
ALLOWED_SYMBOLS = frozenset({"H", "O"})


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / "small_clusters"


def is_water_only_cluster(cluster, min_atoms: int, max_atoms: int) -> bool:
    symbols = cluster.get_chemical_symbols()
    return min_atoms <= len(symbols) <= max_atoms and set(symbols).issubset(ALLOWED_SYMBOLS)


def water_cluster_candidates(task):
    frame_index, atoms, cutoff, bond_fct, vacuum, min_atoms, max_atoms, target_atoms = task
    atoms, mol_indices, centers = base.wrapped_molecules(atoms, bond_fct)
    lengths = atoms.cell.lengths()
    center_distances = np.linalg.norm(base.minimum_image(centers - 0.5 * lengths, lengths), axis=1)
    candidates = []

    for center_id in np.argsort(center_distances):
        cluster, selected_molecules = base.cluster_for_center(
            atoms, mol_indices, centers, int(center_id), cutoff, vacuum
        )
        if not is_water_only_cluster(cluster, min_atoms, max_atoms):
            continue
        cluster.info["selected_molecules"] = ",".join(str(i) for i in selected_molecules)
        cluster.info["interaction_cutoff_A"] = cutoff
        candidates.append(
            (
                abs(len(cluster) - target_atoms),
                float(center_distances[int(center_id)]),
                frame_index,
                cluster,
                int(center_id),
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(frame_index, cluster, center_id) for _, _, frame_index, cluster, center_id in candidates]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cutoff-defined pure H/O water clusters.")
    parser.add_argument("--run-dir", type=Path, default=base.DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cutoff-ps", type=float, default=25.0, help="Use frames at/after this time.")
    parser.add_argument(
        "--interaction-cutoff",
        type=float,
        default=DEFAULT_INTERACTION_CUTOFF,
        help="Atom-atom cutoff for including whole molecules.",
    )
    parser.add_argument("--clusters", type=int, default=30, help="Number of accepted H/O-only clusters to save.")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for cluster extraction candidates.")
    parser.add_argument("--bond-fct", type=float, default=DEFAULT_BOND_FCT, help="aseMolec molecular connectivity scale.")
    parser.add_argument("--target-atoms", type=int, default=DEFAULT_TARGET_ATOMS)
    parser.add_argument("--min-atoms", type=int, default=DEFAULT_MIN_ATOMS)
    parser.add_argument("--max-atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for cluster extraction.")
    args = parser.parse_args()
    if args.min_atoms > args.max_atoms:
        raise ValueError("--min-atoms cannot exceed --max-atoms")

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.run_dir)
    workers = max(1, args.workers)

    base.status(f"Finding trajectory in {args.run_dir}")
    xyz = base.find_xyz(args.run_dir)
    base.status(f"Reading trajectory: {xyz}")
    frames = read(xyz, ":")
    base.status(f"Loaded {len(frames)} frames")
    base.status("Reading frame times")
    times_ps = base.frame_times_ps(args.run_dir, len(frames))
    cluster_candidates = base.production_indices(times_ps, args.cutoff_ps, 0, args.stride)
    base.status(
        f"Searching {len(cluster_candidates)} post-cutoff frames from "
        f"{times_ps[int(cluster_candidates[0])]:.6g} to {times_ps[int(cluster_candidates[-1])]:.6g} ps"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = args.output_dir / "clusters"
    cluster_dir.mkdir(exist_ok=True)
    run_name = args.run_dir.name
    stale_clusters = list(cluster_dir.glob(f"{run_name}_small_cluster_*.xyz"))
    stale_clusters.extend(cluster_dir.glob(f"{run_name}_water_cluster_*.xyz"))
    if stale_clusters:
        base.status(f"Removing {len(stale_clusters)} existing small cluster files for {run_name}")
    for old_cluster in stale_clusters:
        old_cluster.unlink()

    summary_path = args.output_dir / "small_cluster_summary.csv"
    sizes = []
    progress_step = max(1, len(cluster_candidates) // 20)
    base.status(
        f"Searching for {args.clusters} complete cutoff-defined H/O-only clusters "
        f"with {args.min_atoms}-{args.max_atoms} atoms using {workers} workers"
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id", "frame", "time_ps", "natoms", "formula",
                "n_O", "n_N", "n_H", "center_molecule", "interaction_cutoff_A",
                "target_atoms", "min_atoms", "max_atoms",
                "selected_molecules",
            ],
        )
        writer.writeheader()

        tasks = (
            (
                int(frame_index),
                frames[int(frame_index)],
                args.interaction_cutoff,
                args.bond_fct,
                args.vacuum,
                args.min_atoms,
                args.max_atoms,
                args.target_atoms,
            )
            for frame_index in cluster_candidates
        )
        if workers <= 1 or len(cluster_candidates) <= 1:
            results = map(water_cluster_candidates, tasks)
            executor = None
        else:
            chunksize = max(1, len(cluster_candidates) // (workers * 8))
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(water_cluster_candidates, tasks, chunksize=chunksize)

        try:
            cluster_no = 0
            checked_frames = 0
            saved_keys = set()
            for checked_frames, frame_results in enumerate(results, start=1):
                if checked_frames == len(cluster_candidates) or checked_frames % progress_step == 0:
                    base.status(
                        f"Water cluster search: {checked_frames}/{len(cluster_candidates)} frames checked, "
                        f"{cluster_no}/{args.clusters} saved"
                    )
                for frame_index, cluster, center_id in frame_results:
                    if cluster_no >= args.clusters:
                        break
                    selected_key = (int(frame_index), cluster.info.get("selected_molecules", ""))
                    if selected_key in saved_keys:
                        continue
                    saved_keys.add(selected_key)
                    cluster_no += 1
                    sizes.append(len(cluster))
                    cluster.info.update({
                        "source_xyz": str(xyz),
                        "source_frame": int(frame_index),
                        "source_time_ps": float(times_ps[int(frame_index)]),
                        "center_molecule": center_id,
                    })
                    symbols = cluster.get_chemical_symbols()
                    out = cluster_dir / f"{run_name}_water_cluster_{cluster_no:03d}.xyz"
                    write(out, cluster)
                    writer.writerow({
                        "cluster_id": cluster_no,
                        "frame": int(frame_index),
                        "time_ps": f"{times_ps[int(frame_index)]:.6g}",
                        "natoms": len(cluster),
                        "formula": cluster.get_chemical_formula(),
                        "n_O": symbols.count("O"),
                        "n_N": symbols.count("N"),
                        "n_H": symbols.count("H"),
                        "center_molecule": center_id,
                        "interaction_cutoff_A": (
                            f"{cluster.info.get('interaction_cutoff_A', args.interaction_cutoff):.6g}"
                        ),
                        "target_atoms": args.target_atoms,
                        "min_atoms": args.min_atoms,
                        "max_atoms": args.max_atoms,
                        "selected_molecules": cluster.info.get("selected_molecules", ""),
                    })
                if cluster_no >= args.clusters:
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    sizes = np.array(sizes)
    print(f"Input trajectory: {xyz}")
    print(f"Saved clusters: {cluster_dir}")
    if len(sizes) == 0:
        raise RuntimeError("No pure H/O water clusters were extracted.")
    if len(sizes) < args.clusters:
        raise RuntimeError(
            f"Only found {len(sizes)} pure H/O clusters with {args.min_atoms}-{args.max_atoms} atoms; "
            f"requested {args.clusters}."
        )
    print(f"Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
