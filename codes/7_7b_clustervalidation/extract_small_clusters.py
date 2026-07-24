from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from ase.io import read, write

import extract_large_clusters as base

DEFAULT_PREFERRED_ATOMS = 12
DEFAULT_INTERACTION_CUTOFF = 1.75
DEFAULT_CUTOFF_PS = 10.0


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / "small_clusters"


def small_cluster_candidate(task):
    frame_index, atoms, cutoff, bond_scale, preferred_atoms, vacuum = task
    cluster, center_id = base.choose_cluster(atoms, cutoff, bond_scale, preferred_atoms, vacuum)
    return frame_index, cluster, center_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cutoff-defined water/NH3 clusters.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=base.DEFAULT_RUN,
        help=f"Trajectory output directory to process. Default: outputsfull/{base.DEFAULT_DATA_SOURCE_NAME}",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cutoff-ps", type=float, default=DEFAULT_CUTOFF_PS, help="Use frames at/after this time.")
    parser.add_argument(
        "--interaction-cutoff",
        type=float,
        default=DEFAULT_INTERACTION_CUTOFF,
        help="Atom-atom cutoff for including whole molecules.",
    )
    parser.add_argument("--clusters", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for cluster extraction candidates.")
    parser.add_argument(
        "--bond-scale",
        type=float,
        default=base.DEFAULT_BOND_SCALE,
        help="ASE covalent-radius multiplier for molecule detection.",
    )
    parser.add_argument(
        "--preferred-atoms",
        type=int,
        default=DEFAULT_PREFERRED_ATOMS,
        help="Preferred cluster size used to choose the center molecule; clusters are not rejected by size.",
    )
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for cluster extraction.")
    args = parser.parse_args()
    if args.bond_scale <= 0:
        parser.error("--bond-scale must be positive")

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
    cluster_candidates = base.production_indices(times_ps, args.cutoff_ps, args.clusters, args.stride)
    base.status(
        f"Selected {len(cluster_candidates)} cluster candidates from "
        f"{times_ps[int(cluster_candidates[0])]:.6g} to {times_ps[int(cluster_candidates[-1])]:.6g} ps"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_dir.name

    summary_path = args.output_dir / "small_cluster_summary.csv"
    clusters_path = args.output_dir / f"{run_name}_small_clusters.xyz"
    sizes = []
    clusters = []
    progress_step = max(1, len(cluster_candidates) // 20)
    base.status(f"Extracting {len(cluster_candidates)} small clusters using {workers} workers")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id", "frame", "time_ps", "natoms", "formula",
                "n_O", "n_N", "n_H", "center_molecule", "interaction_cutoff_A",
                "selected_molecules",
            ],
        )
        writer.writeheader()

        tasks = (
            (
                int(frame_index),
                frames[int(frame_index)],
                args.interaction_cutoff,
                args.bond_scale,
                args.preferred_atoms,
                args.vacuum,
            )
            for frame_index in cluster_candidates
        )
        if workers <= 1 or len(cluster_candidates) <= 1:
            results = map(small_cluster_candidate, tasks)
            executor = None
        else:
            chunksize = max(1, len(cluster_candidates) // (workers * 8))
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(small_cluster_candidate, tasks, chunksize=chunksize)

        try:
            for cluster_no, (frame_index, cluster, center_id) in enumerate(results, start=1):
                if cluster_no == len(cluster_candidates) or cluster_no % progress_step == 0:
                    base.status(f"Small cluster progress: {cluster_no}/{len(cluster_candidates)}")
                sizes.append(len(cluster))
                cluster.info.update({
                    "cluster_id": cluster_no,
                    "source_xyz": str(xyz),
                    "source_frame": int(frame_index),
                    "source_time_ps": float(times_ps[int(frame_index)]),
                    "center_molecule": center_id,
                })
                symbols = cluster.get_chemical_symbols()
                clusters.append(cluster)
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
                    "selected_molecules": cluster.info.get("selected_molecules", ""),
                })
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    sizes = np.array(sizes)
    print(f"Input trajectory: {xyz}")
    if len(sizes) == 0:
        raise RuntimeError("No small clusters were extracted.")
    write(clusters_path, clusters)
    print(f"Saved clusters: {clusters_path}")
    print(f"Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}")
    print(f"Clusters with {args.preferred_atoms} atoms: {np.count_nonzero(sizes == args.preferred_atoms)}/{len(sizes)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
