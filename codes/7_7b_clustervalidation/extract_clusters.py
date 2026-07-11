from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUN = REPO_ROOT / "outputsfull" / "r09_hot_w7n1"


def import_anaatoms():
    if "ase_ga.utilities" not in sys.modules:
        ase_ga = types.ModuleType("ase_ga")
        utilities = types.ModuleType("ase_ga.utilities")
        utilities.get_rdf = lambda *args, **kwargs: None
        ase_ga.utilities = utilities
        sys.modules["ase_ga"] = ase_ga
        sys.modules["ase_ga.utilities"] = utilities
    sys.path.insert(0, str(REPO_ROOT / "aseMolec"))
    from aseMolec import anaAtoms

    return anaAtoms


anaAtoms = import_anaatoms()


def status(message: str) -> None:
    print(message, flush=True)


def find_xyz(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("*.xyz"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not files:
        raise FileNotFoundError(f"No .xyz trajectory found in {run_dir}")
    return files[0]


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / "cluster_validation"


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


def wrapped_molecules(atoms: Atoms, bond_fct: float) -> tuple[Atoms, list[np.ndarray], np.ndarray]:
    atoms = atoms.copy()
    anaAtoms.find_molecs([atoms], fct=bond_fct)
    mol_ids = atoms.arrays["molID"]
    indices: list[np.ndarray] = []
    centers = []
    for mol_id in np.unique(mol_ids):
        idx = np.flatnonzero(mol_ids == mol_id)
        mol = atoms[idx]
        center = anaAtoms.wrap_molec(mol, fct=bond_fct)
        atoms.positions[idx] = mol.positions
        indices.append(idx)
        centers.append(center)
    return atoms, indices, np.array(centers)


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


def choose_cluster(atoms: Atoms, cutoff: float, bond_fct: float, target: tuple[int, int], vacuum: float) -> tuple[Atoms, int]:
    atoms, mol_indices, centers = wrapped_molecules(atoms, bond_fct)
    lengths = atoms.cell.lengths()
    order = np.argsort(np.linalg.norm(minimum_image(centers - 0.5 * lengths, lengths), axis=1))
    best = None
    best_score = float("inf")

    for center_id in order:
        cluster, selected_molecules = cluster_for_center(atoms, mol_indices, centers, int(center_id), cutoff, vacuum)
        cluster.info["selected_molecules"] = ",".join(str(i) for i in selected_molecules)
        natoms = len(cluster)
        if target[0] <= natoms <= target[1]:
            return cluster, int(center_id)
        score = min(abs(natoms - target[0]), abs(natoms - target[1]))
        if score < best_score:
            best = (cluster, int(center_id))
            best_score = score
    if best is None:
        raise RuntimeError("No cluster could be extracted.")
    return best


def rdf_pairs(frames: list[Atoms]) -> list[tuple[str, str]]:
    symbols = sorted({symbol for atoms in frames for symbol in atoms.get_chemical_symbols()})
    preferred = ["O", "N", "H"]
    symbols = [sym for sym in preferred if sym in symbols] + [sym for sym in symbols if sym not in preferred]
    return [(a, b) for i, a in enumerate(symbols) for b in symbols[i:]]


def rdf_for_frame(task: tuple[Atoms, float, int, list[tuple[str, str]]]) -> dict[str, np.ndarray]:
    atoms, rmax, nbins, pairs = task
    edges = np.linspace(0.0, rmax, nbins + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 * np.pi * radii**2 * np.diff(edges)
    accum = {f"{a}-{b}": np.zeros(nbins) for a, b in pairs}

    symbols = np.array(atoms.get_chemical_symbols())
    volume = atoms.get_volume()
    distances = atoms.get_all_distances(mic=True)
    for a, b in pairs:
        ia = np.flatnonzero(symbols == a)
        ib = np.flatnonzero(symbols == b)
        if not len(ia) or not len(ib):
            continue
        if a == b:
            d = distances[np.ix_(ia, ia)]
            d = d[np.triu_indices_from(d, k=1)]
            norm = 0.5 * len(ia) * (len(ib) / volume) * shell
        else:
            d = distances[np.ix_(ia, ib)].ravel()
            norm = len(ia) * (len(ib) / volume) * shell
        hist, _ = np.histogram(d[(d > 0.0) & (d < rmax)], bins=edges)
        accum[f"{a}-{b}"] += hist / np.maximum(norm, 1e-30)

    return accum


def rdf_for_frames(
    frames: list[Atoms],
    rmax: float,
    nbins: int,
    pairs: list[tuple[str, str]],
    workers: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    edges = np.linspace(0.0, rmax, nbins + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    accum = {f"{a}-{b}": np.zeros(nbins) for a, b in pairs}
    total = len(frames)
    progress_step = max(1, total // 20)

    if workers <= 1 or total <= 1:
        for done, atoms in enumerate(frames, start=1):
            frame_rdfs = rdf_for_frame((atoms, rmax, nbins, pairs))
            for key, rdf in frame_rdfs.items():
                accum[key] += rdf
            if done == total or done % progress_step == 0:
                status(f"RDF progress: {done}/{total} frames")
    else:
        chunksize = max(1, total // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            tasks = ((atoms, rmax, nbins, pairs) for atoms in frames)
            for done, frame_rdfs in enumerate(executor.map(rdf_for_frame, tasks, chunksize=chunksize), start=1):
                for key, rdf in frame_rdfs.items():
                    accum[key] += rdf
                if done == total or done % progress_step == 0:
                    status(f"RDF progress: {done}/{total} frames")

    for key in accum:
        accum[key] /= total
    return r, accum


def cluster_candidate(task: tuple[int, Atoms, float, float, tuple[int, int], float]) -> tuple[int, Atoms | None, int | None]:
    frame_index, atoms, cutoff, bond_fct, target, vacuum = task
    cluster, center_id = choose_cluster(atoms, cutoff, bond_fct, target, vacuum)
    if not (target[0] <= len(cluster) <= target[1]):
        return frame_index, None, None
    return frame_index, cluster, center_id


def plot_rdfs(path: Path, r: np.ndarray, rdfs: dict[str, np.ndarray], cutoff: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, rdf in rdfs.items():
        ax.plot(r, rdf, label=key)
    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.0, label=f"cluster cutoff {cutoff:g} A")
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RDFs and extract DFT-sized water/NH3 clusters.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cutoff-ps", type=float, default=25.0, help="Use frames at/after this time.")
    parser.add_argument("--interaction-cutoff", type=float, default=2.0, help="Atom-atom cutoff for including whole molecules.")
    parser.add_argument("--clusters", type=int, default=30)
    parser.add_argument("--rdf-frames", type=int, default=0, help="Maximum RDF frames to use after cutoff; 0 uses all.")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for cluster extraction candidates.")
    parser.add_argument("--rmax", type=float, default=6.0)
    parser.add_argument("--nbins", type=int, default=160)
    parser.add_argument("--bond-fct", type=float, default=1.0, help="aseMolec molecular connectivity scale.")
    parser.add_argument("--target-min-atoms", type=int, default=10)
    parser.add_argument("--target-max-atoms", type=int, default=13)
    parser.add_argument("--vacuum", type=float, default=24.0)
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for RDF and cluster extraction.")
    args = parser.parse_args()
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
    rdf_idx = production_indices(times_ps, args.cutoff_ps, args.rdf_frames)
    cluster_candidates = production_indices(times_ps, args.cutoff_ps, 0, args.stride)
    status(
        f"Selected {len(rdf_idx)} RDF frames after {args.cutoff_ps:g} ps "
        f"and {len(cluster_candidates)} cluster candidates with stride {args.stride}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = args.output_dir / "clusters"
    cluster_dir.mkdir(exist_ok=True)
    run_name = args.run_dir.name
    stale_clusters = list(cluster_dir.glob(f"{run_name}_cluster_*.xyz"))
    if stale_clusters:
        status(f"Removing {len(stale_clusters)} existing cluster files for {run_name}")
    for old_cluster in cluster_dir.glob(f"{run_name}_cluster_*.xyz"):
        old_cluster.unlink()

    rdf_frames = [frames[int(i)] for i in rdf_idx]
    pairs = rdf_pairs(rdf_frames)
    status(f"Computing RDFs for pairs {', '.join(f'{a}-{b}' for a, b in pairs)} using {workers} workers")
    r, rdfs = rdf_for_frames(rdf_frames, args.rmax, args.nbins, pairs, workers)
    rdf_path = args.output_dir / f"{run_name}_rdf.png"
    status(f"Saving RDF plot: {rdf_path}")
    plot_rdfs(rdf_path, r, rdfs, args.interaction_cutoff)

    summary_path = args.output_dir / "cluster_summary.csv"
    sizes = []
    target = (args.target_min_atoms, args.target_max_atoms)
    cluster_progress_step = max(1, len(cluster_candidates) // 20)
    status(f"Extracting up to {args.clusters} clusters using {workers} workers")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id", "frame", "time_ps", "natoms", "formula",
                "n_O", "n_N", "n_H", "center_molecule", "selected_molecules", "good_size",
            ],
        )
        writer.writeheader()
        cluster_no = 0

        cluster_tasks = (
            (
                int(frame_index),
                frames[int(frame_index)],
                args.interaction_cutoff,
                args.bond_fct,
                target,
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

        found_enough = False
        try:
            for checked, (frame_index, cluster, center_id) in enumerate(cluster_results, start=1):
                if checked == len(cluster_candidates) or checked % cluster_progress_step == 0:
                    status(
                        f"Cluster progress: {checked}/{len(cluster_candidates)} candidates, "
                        f"{cluster_no}/{args.clusters} accepted"
                    )
                if cluster is None or center_id is None:
                    continue
                cluster_no += 1
                sizes.append(len(cluster))
                cluster.info.update({
                    "source_xyz": str(xyz),
                    "source_frame": int(frame_index),
                    "source_time_ps": float(times_ps[int(frame_index)]),
                    "interaction_cutoff_A": args.interaction_cutoff,
                    "center_molecule": center_id,
                })
                symbols = cluster.get_chemical_symbols()
                out = cluster_dir / f"{run_name}_cluster_{cluster_no:03d}.xyz"
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
                    "selected_molecules": cluster.info.get("selected_molecules", ""),
                    "good_size": args.target_min_atoms <= len(cluster) <= args.target_max_atoms,
                })
                status(f"Accepted cluster {cluster_no}/{args.clusters} from frame {frame_index}")
                if cluster_no >= args.clusters:
                    found_enough = True
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=found_enough)

    sizes = np.array(sizes)
    print(f"Input trajectory: {xyz}")
    print(f"RDF frames: {len(rdf_frames)} after {args.cutoff_ps:g} ps")
    print(f"RDF pairs: {', '.join(rdfs)}")
    print(f"Saved RDF plot: {rdf_path}")
    print(f"Saved clusters: {cluster_dir}")
    if len(sizes) == 0:
        raise RuntimeError(
            f"No clusters found in the {args.target_min_atoms}-{args.target_max_atoms} atom target range. "
            "Try increasing --interaction-cutoff or reducing --stride."
        )
    print(f"Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}")
    print(f"Good-size clusters ({args.target_min_atoms}-{args.target_max_atoms} atoms): "
          f"{np.count_nonzero((sizes >= args.target_min_atoms) & (sizes <= args.target_max_atoms))}/{len(sizes)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
