from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_SOURCE_NAME = "r09_hot_w"
DEFAULT_RUN = REPO_ROOT / "outputsfull" / DEFAULT_DATA_SOURCE_NAME
DEFAULT_INTERACTION_CUTOFF = 2.15
DEFAULT_BOND_SCALE = 1.2
DEFAULT_PREFERRED_ATOMS = 21
DEFAULT_CUTOFF_PS = 10.0


def status(message: str) -> None:
    print(message, flush=True)


def find_xyz(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("*.xyz"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not files:
        raise FileNotFoundError(f"No .xyz trajectory found in {run_dir}")
    return files[0]


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / "large_clusters"


def frame_times_ps(run_dir: Path, n_frames: int) -> np.ndarray:
    txts = sorted(run_dir.glob("*thermo*.txt"))
    if txts:
        header = txts[0].read_text(encoding="utf-8", errors="ignore").splitlines()[0].lstrip("#").split()
        data = np.atleast_2d(np.loadtxt(txts[0]))
        if "time_fs" in header and len(data) >= n_frames:
            return data[:n_frames, header.index("time_fs")] / 1000.0
    return np.arange(n_frames, dtype=float)


def production_indices(times_ps: np.ndarray, cutoff_ps: float, count: int, stride: int = 1) -> np.ndarray:
    available = np.flatnonzero(times_ps >= cutoff_ps)
    if available.size == 0:
        raise ValueError(f"Cutoff {cutoff_ps:g} ps leaves no frames.")
    available = available[:: max(1, stride)]
    if count > 0 and available.size > count:
        available = available[np.round(np.linspace(0, available.size - 1, count)).astype(int)]
    return available


def minimum_image(delta: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return delta - np.rint(delta / lengths) * lengths


def wrapped_molecules(atoms: Atoms, bond_scale: float) -> tuple[Atoms, list[np.ndarray], np.ndarray]:
    atoms = atoms.copy()
    cutoffs = natural_cutoffs(atoms, mult=bond_scale)
    neighbors = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    neighbors.update(atoms)
    cell = atoms.cell.array
    visited: set[int] = set()
    molecule_indices: list[np.ndarray] = []
    centers = []
    wrapped_positions = atoms.positions.copy()

    for seed in range(len(atoms)):
        if seed in visited:
            continue
        component: list[int] = []
        unwrapped: dict[int, np.ndarray] = {seed: atoms.positions[seed].copy()}
        stack = [seed]
        visited.add(seed)

        while stack:
            atom_index = stack.pop()
            component.append(atom_index)
            indices, offsets = neighbors.get_neighbors(atom_index)
            current_shift = unwrapped[atom_index] - atoms.positions[atom_index]
            for neighbor_index, offset in zip(indices, offsets):
                neighbor_index = int(neighbor_index)
                if neighbor_index in visited:
                    continue
                unwrapped[neighbor_index] = atoms.positions[neighbor_index] + np.asarray(offset) @ cell + current_shift
                visited.add(neighbor_index)
                stack.append(neighbor_index)

        idx = np.array(sorted(component), dtype=int)
        positions = np.array([unwrapped[int(atom_index)] for atom_index in idx])
        wrapped_positions[idx] = positions
        molecule_indices.append(idx)
        centers.append(positions.mean(axis=0))

    atoms.positions = wrapped_positions
    if not molecule_indices:
        raise RuntimeError("No molecules were identified.")
    return atoms, molecule_indices, np.array(centers)


def cluster_for_center(
    atoms: Atoms,
    molecule_indices: list[np.ndarray],
    centers: np.ndarray,
    center_id: int,
    cutoff: float,
    vacuum: float,
) -> tuple[Atoms, list[int]]:
    lengths = atoms.cell.lengths()
    center_pos = atoms.positions[molecule_indices[center_id]]
    selected_positions = []
    selected_symbols = []
    selected_molecules = []

    for mol_id, idx in enumerate(molecule_indices):
        shift = minimum_image(centers[mol_id] - centers[center_id], lengths)
        pos = atoms.positions[idx] + (centers[center_id] + shift - centers[mol_id])
        if np.linalg.norm(pos[:, None, :] - center_pos[None, :, :], axis=2).min() <= cutoff:
            selected_positions.extend(pos.tolist())
            selected_symbols.extend(np.array(atoms.get_chemical_symbols())[idx].tolist())
            selected_molecules.append(mol_id)

    cluster = Atoms(symbols=selected_symbols, positions=np.array(selected_positions), pbc=False)
    cluster.set_cell([vacuum, vacuum, vacuum])
    cluster.positions += 0.5 * vacuum - cluster.get_center_of_mass()
    return cluster, selected_molecules


def choose_cluster(
    atoms: Atoms,
    cutoff: float,
    bond_scale: float,
    preferred_atoms: int,
    vacuum: float,
) -> tuple[Atoms, int]:
    atoms, mol_indices, centers = wrapped_molecules(atoms, bond_scale)
    lengths = atoms.cell.lengths()
    order = np.argsort(np.linalg.norm(minimum_image(centers - 0.5 * lengths, lengths), axis=1))
    best = None
    best_score = float("inf")

    for center_id in order:
        cluster, selected_molecules = cluster_for_center(atoms, mol_indices, centers, int(center_id), cutoff, vacuum)
        cluster.info["selected_molecules"] = ",".join(str(i) for i in selected_molecules)
        cluster.info["interaction_cutoff_A"] = cutoff
        score = abs(len(cluster) - preferred_atoms)
        if score < best_score:
            best = (cluster, int(center_id))
            best_score = score
    if best is None:
        raise RuntimeError("No cluster could be extracted.")
    return best


def cluster_candidate(
    task: tuple[int, Atoms, float, float, int, float],
) -> tuple[int, Atoms, int]:
    frame_index, atoms, cutoff, bond_scale, preferred_atoms, vacuum = task
    cluster, center_id = choose_cluster(atoms, cutoff, bond_scale, preferred_atoms, vacuum)
    return frame_index, cluster, center_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract large cutoff-defined water/NH3 clusters.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN,
        help=f"Trajectory output directory to process. Default: outputsfull/{DEFAULT_DATA_SOURCE_NAME}",
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
        default=DEFAULT_BOND_SCALE,
        help="ASE covalent-radius multiplier for molecule detection.",
    )
    parser.add_argument(
        "--preferred-atoms",
        type=int,
        default=DEFAULT_PREFERRED_ATOMS,
        help="Preferred cluster size used only to choose the center molecule; clusters are not rejected by size.",
    )
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for cluster extraction.")
    args = parser.parse_args()
    if args.bond_scale <= 0:
        parser.error("--bond-scale must be positive")
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.run_dir)
    workers = max(1, args.workers)

    status(f"Finding trajectory in {args.run_dir}")
    xyz = find_xyz(args.run_dir)
    status(f"Reading trajectory: {xyz}")
    frames = read(xyz, ":")
    status(f"Loaded {len(frames)} frames")
    status("Reading frame times")
    times_ps = frame_times_ps(args.run_dir, len(frames))
    cluster_candidates = production_indices(times_ps, args.cutoff_ps, args.clusters, args.stride)
    status(
        f"Selected {len(cluster_candidates)} cluster candidates from "
        f"{times_ps[int(cluster_candidates[0])]:.6g} to {times_ps[int(cluster_candidates[-1])]:.6g} ps"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_dir.name

    summary_path = args.output_dir / "large_cluster_summary.csv"
    clusters_path = args.output_dir / f"{run_name}_large_clusters.xyz"
    sizes = []
    clusters = []
    cluster_progress_step = max(1, len(cluster_candidates) // 20)
    status(f"Extracting {len(cluster_candidates)} large clusters using {workers} workers")
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
        cluster_no = 0

        cluster_tasks = (
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
            cluster_results = map(cluster_candidate, cluster_tasks)
            executor = None
        else:
            chunksize = max(1, len(cluster_candidates) // (workers * 8))
            executor = ProcessPoolExecutor(max_workers=workers)
            cluster_results = executor.map(cluster_candidate, cluster_tasks, chunksize=chunksize)

        try:
            for checked, (frame_index, cluster, center_id) in enumerate(cluster_results, start=1):
                if checked == len(cluster_candidates) or checked % cluster_progress_step == 0:
                    status(
                        f"Cluster progress: {checked}/{len(cluster_candidates)} candidates, "
                        f"{cluster_no}/{args.clusters} saved"
                    )
                cluster_no += 1
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
                status(f"Prepared large cluster {cluster_no}/{args.clusters} from frame {frame_index}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    sizes = np.array(sizes)
    print(f"Input trajectory: {xyz}")
    if len(sizes) == 0:
        raise RuntimeError("No clusters were extracted.")
    write(clusters_path, clusters)
    print(f"Saved large clusters: {clusters_path}")
    print(f"Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}")
    print(f"Clusters with {args.preferred_atoms} atoms: {np.count_nonzero(sizes == args.preferred_atoms)}/{len(sizes)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
