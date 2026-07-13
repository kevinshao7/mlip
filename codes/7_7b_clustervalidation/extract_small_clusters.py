from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from ase.io import read, write

import extract_clusters as base


DEFAULT_INTERACTION_CUTOFF = 1.7
DEFAULT_BOND_FCT = 1.0
DEFAULT_PREFERRED_ATOMS = 12


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / "small_clusters"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract small DFT-sized water/NH3 clusters.")
    parser.add_argument("--run-dir", type=Path, default=base.DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cutoff-ps", type=float, default=25.0, help="Use frames at/after this time.")
    parser.add_argument(
        "--interaction-cutoff",
        type=float,
        default=DEFAULT_INTERACTION_CUTOFF,
        help="Atom-atom cutoff for including whole molecules.",
    )
    parser.add_argument("--clusters", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for cluster extraction candidates.")
    parser.add_argument("--bond-fct", type=float, default=DEFAULT_BOND_FCT, help="aseMolec molecular connectivity scale.")
    parser.add_argument(
        "--preferred-atoms",
        type=int,
        default=DEFAULT_PREFERRED_ATOMS,
        help="Preferred cluster size used only to choose the center molecule; clusters are not rejected by size.",
    )
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for cluster extraction.")
    args = parser.parse_args()

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
    cluster_dir = args.output_dir / "clusters"
    cluster_dir.mkdir(exist_ok=True)
    run_name = args.run_dir.name
    stale_clusters = list(cluster_dir.glob(f"{run_name}_small_cluster_*.xyz"))
    if stale_clusters:
        base.status(f"Removing {len(stale_clusters)} existing small cluster files for {run_name}")
    for old_cluster in stale_clusters:
        old_cluster.unlink()

    summary_path = args.output_dir / "small_cluster_summary.csv"
    sizes = []
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
                args.bond_fct,
                args.preferred_atoms,
                args.vacuum,
            )
            for frame_index in cluster_candidates
        )
        if workers <= 1 or len(cluster_candidates) <= 1:
            results = map(base.cluster_candidate, tasks)
            executor = None
        else:
            chunksize = max(1, len(cluster_candidates) // (workers * 8))
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(base.cluster_candidate, tasks, chunksize=chunksize)

        try:
            for cluster_no, (frame_index, cluster, center_id) in enumerate(results, start=1):
                if cluster_no == len(cluster_candidates) or cluster_no % progress_step == 0:
                    base.status(f"Small cluster progress: {cluster_no}/{len(cluster_candidates)}")
                sizes.append(len(cluster))
                cluster.info.update({
                    "source_xyz": str(xyz),
                    "source_frame": int(frame_index),
                    "source_time_ps": float(times_ps[int(frame_index)]),
                    "center_molecule": center_id,
                })
                symbols = cluster.get_chemical_symbols()
                out = cluster_dir / f"{run_name}_small_cluster_{cluster_no:03d}.xyz"
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
                    "selected_molecules": cluster.info.get("selected_molecules", ""),
                })
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    sizes = np.array(sizes)
    print(f"Input trajectory: {xyz}")
    print(f"Saved clusters: {cluster_dir}")
    if len(sizes) == 0:
        raise RuntimeError("No small clusters were extracted.")
    print(f"Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}")
    print(f"Clusters with 10-13 atoms: {np.count_nonzero((sizes >= 10) & (sizes <= 13))}/{len(sizes)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
