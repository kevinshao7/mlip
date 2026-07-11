from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUN = REPO_ROOT / "outputsfull" / "r09_hot_w7n1"


def status(message: str) -> None:
    print(message, flush=True)


def find_xyz(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("*.xyz"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not files:
        raise FileNotFoundError(f"No .xyz trajectory found in {run_dir}")
    return files[0]


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
    parser = argparse.ArgumentParser(description="Calculate RDFs for an equilibrated trajectory window.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cutoff-ps", type=float, default=25.0, help="Use frames at/after this time.")
    parser.add_argument("--rdf-frames", type=int, default=0, help="Maximum RDF frames to use after cutoff; 0 uses all.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rmax", type=float, default=6.0)
    parser.add_argument("--nbins", type=int, default=160)
    parser.add_argument("--interaction-cutoff", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)

    status(f"Finding trajectory in {args.run_dir}")
    xyz = find_xyz(args.run_dir)
    status(f"Reading trajectory: {xyz}")
    frames = read(xyz, ":")
    status(f"Loaded {len(frames)} frames")
    times_ps = frame_times_ps(args.run_dir, len(frames))
    rdf_idx = production_indices(times_ps, args.cutoff_ps, args.rdf_frames, args.stride)
    status(
        f"Selected {len(rdf_idx)} RDF frames from "
        f"{times_ps[int(rdf_idx[0])]:.6g} to {times_ps[int(rdf_idx[-1])]:.6g} ps"
    )

    rdf_frames = [frames[int(i)] for i in rdf_idx]
    pairs = rdf_pairs(rdf_frames)
    status(f"Computing RDFs for pairs {', '.join(f'{a}-{b}' for a, b in pairs)} using {workers} workers")
    r, rdfs = rdf_for_frames(rdf_frames, args.rmax, args.nbins, pairs, workers)

    rdf_path = output_dir / f"{args.run_dir.name}_rdf.png"
    status(f"Saving RDF plot: {rdf_path}")
    plot_rdfs(rdf_path, r, rdfs, args.interaction_cutoff)
    print(f"Input trajectory: {xyz}")
    print(f"RDF frames: {len(rdf_frames)} after {args.cutoff_ps:g} ps")
    print(f"RDF pairs: {', '.join(rdfs)}")
    print(f"Saved RDF plot: {rdf_path}")


if __name__ == "__main__":
    main()
