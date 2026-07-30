from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

# Parallel usage:
#   python compute_trajectory_rdf.py --cores=24
#
# Use process-level parallelism for RDF frames. Keep OMP/MKL/OpenBLAS threads at
# 1 so --cores=24 means 24 Python worker processes, not 24 workers times many
# BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputsfull" / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read


DEFAULT_RUN = REPO_ROOT / "outputsfull" / "r09_hot_w"
DEFAULT_XYZ = DEFAULT_RUN / "pressure_equil_seed_353168294_P_15GPa_T_300K_density_0.2.xyz"
DEFAULT_THERMO = DEFAULT_RUN / "pressure_equil_seed_353168294_P_15GPa_T_300K_density_0.2_thermo.txt"


def status(message: str) -> None:
    print(message, flush=True)


def find_xyz(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("*.xyz"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    files = [path for path in files if "checkpoint" not in path.name.lower()] or files
    if not files:
        raise FileNotFoundError(f"No .xyz trajectory found in {run_dir}")
    return files[0]


def matching_thermo_path(xyz_path: Path, thermo_path: Path | None) -> Path | None:
    if thermo_path is not None:
        return thermo_path
    candidate = xyz_path.with_name(f"{xyz_path.stem}_thermo.txt")
    if candidate.is_file():
        return candidate
    txts = sorted(xyz_path.parent.glob("*thermo*.txt"))
    return txts[0] if txts else None


def frame_times_ps(xyz_path: Path, thermo_path: Path | None, n_frames: int) -> tuple[np.ndarray, Path | None]:
    selected_thermo = matching_thermo_path(xyz_path, thermo_path)
    if selected_thermo is not None:
        header = selected_thermo.read_text(encoding="utf-8", errors="ignore").splitlines()[0].lstrip("#").split()
        data = np.atleast_2d(np.loadtxt(selected_thermo))
        if "time_fs" in header and len(data) >= n_frames:
            return data[:n_frames, header.index("time_fs")] / 1000.0, selected_thermo
        raise ValueError(
            f"{selected_thermo} must contain a time_fs column and at least {n_frames} rows; "
            f"found columns {header} and {len(data)} row(s)."
        )
    return np.arange(n_frames, dtype=float), None


def production_indices(
    times_ps: np.ndarray,
    equilibration_fraction: float,
    cutoff_ps: float | None,
    count: int,
    stride: int = 1,
) -> np.ndarray:
    if not 0.0 <= equilibration_fraction < 1.0:
        raise ValueError("--equilibration-fraction must be in [0, 1).")
    start_index = int(np.ceil(len(times_ps) * equilibration_fraction))
    available = np.arange(start_index, len(times_ps), dtype=int)
    if cutoff_ps is not None:
        available = available[times_ps[available] >= cutoff_ps]
    if available.size == 0:
        cutoff_text = f" and cutoff {cutoff_ps:g} ps" if cutoff_ps is not None else ""
        raise ValueError(f"Equilibration fraction {equilibration_fraction:g}{cutoff_text} leaves no frames.")
    available = available[:: max(1, stride)]
    if count > 0 and available.size > count:
        available = available[np.round(np.linspace(0, available.size - 1, count)).astype(int)]
    return available


def validate_rmax(frames: list[Atoms], rmax: float) -> None:
    min_half_length = min(float(np.min(atoms.cell.lengths())) for atoms in frames) / 2.0
    if rmax > min_half_length:
        raise ValueError(
            f"Requested --rmax={rmax:g} A exceeds half the shortest sampled cell length "
            f"({min_half_length:.6g} A). Reduce --rmax for an unambiguous minimum-image RDF."
        )


def rdf_pairs(frames: list[Atoms]) -> list[tuple[str, str]]:
    symbols = sorted({symbol for atoms in frames for symbol in atoms.get_chemical_symbols()})
    preferred = ["O", "N", "H"]
    symbols = [sym for sym in preferred if sym in symbols] + [sym for sym in symbols if sym not in preferred]
    return [(a, b) for i, a in enumerate(symbols) for b in symbols[i:]]


def pair_distances_mic(positions_a: np.ndarray, positions_b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = positions_a[:, None, :] - positions_b[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def rdf_for_frame(task: tuple[Atoms, float, int, list[tuple[str, str]]]) -> dict[str, np.ndarray]:
    atoms, rmax, nbins, pairs = task
    edges = np.linspace(0.0, rmax, nbins + 1)
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.positions
    cell = atoms.cell.array
    volume = atoms.get_volume()
    frame_rdfs: dict[str, np.ndarray] = {}

    for a, b in pairs:
        idx_a = np.flatnonzero(symbols == a)
        idx_b = np.flatnonzero(symbols == b)
        key = f"{a}-{b}"
        if idx_a.size == 0 or idx_b.size == 0:
            frame_rdfs[key] = np.full(nbins, np.nan)
            continue

        distances = pair_distances_mic(positions[idx_a], positions[idx_b], cell)
        if a == b:
            if idx_a.size < 2:
                frame_rdfs[key] = np.full(nbins, np.nan)
                continue
            pair_i, pair_j = np.triu_indices(idx_a.size, k=1)
            distances = distances[pair_i, pair_j]
            hist, _ = np.histogram(distances, bins=edges)
            neighbor_counts = 2.0 * hist
            density_b = (idx_b.size - 1) / volume
        else:
            hist, _ = np.histogram(distances.ravel(), bins=edges)
            neighbor_counts = hist.astype(float)
            density_b = idx_b.size / volume

        denominator = idx_a.size * density_b * shell_volumes
        with np.errstate(divide="ignore", invalid="ignore"):
            frame_rdfs[key] = neighbor_counts / denominator

    return frame_rdfs


def rdf_for_frames(
    frames: list[Atoms],
    rmax: float,
    nbins: int,
    pairs: list[tuple[str, str]],
    cores: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    edges = np.linspace(0.0, rmax, nbins + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    accum = {f"{a}-{b}": np.zeros(nbins) for a, b in pairs}
    counts = {f"{a}-{b}": np.zeros(nbins, dtype=int) for a, b in pairs}
    total = len(frames)
    progress_step = max(1, total // 20)

    def add_frame_rdfs(frame_rdfs: dict[str, np.ndarray]) -> None:
        for key, rdf in frame_rdfs.items():
            finite = np.isfinite(rdf)
            accum[key][finite] += rdf[finite]
            counts[key][finite] += 1

    if cores <= 1 or total <= 1:
        for done, atoms in enumerate(frames, start=1):
            frame_rdfs = rdf_for_frame((atoms, rmax, nbins, pairs))
            add_frame_rdfs(frame_rdfs)
            if done == total or done % progress_step == 0:
                status(f"RDF progress: {done}/{total} frames")
    else:
        chunksize = max(1, total // (cores * 8))
        with ProcessPoolExecutor(max_workers=cores) as executor:
            tasks = ((atoms, rmax, nbins, pairs) for atoms in frames)
            for done, frame_rdfs in enumerate(executor.map(rdf_for_frame, tasks, chunksize=chunksize), start=1):
                add_frame_rdfs(frame_rdfs)
                if done == total or done % progress_step == 0:
                    status(f"RDF progress: {done}/{total} frames")

    for key in accum:
        with np.errstate(divide="ignore", invalid="ignore"):
            accum[key] = accum[key] / counts[key]
    return r, accum


def plot_rdfs(path: Path, r: np.ndarray, rdfs: dict[str, np.ndarray], cutoff: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    finite_values = []
    for key, rdf in rdfs.items():
        finite = np.isfinite(rdf)
        if np.any(finite):
            finite_values.append(rdf[finite])
        ax.plot(r, rdf, label=key)
    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.0, label=f"cluster cutoff {cutoff:g} A")
    ax.set_xlim(float(r[0]), float(r[-1]))
    if finite_values:
        ymax = max(float(np.nanmax(values)) for values in finite_values)
        ax.set_ylim(bottom=0.0, top=max(1.05, 1.08 * ymax))
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_rdf_csv(path: Path, r: np.ndarray, rdfs: dict[str, np.ndarray]) -> Path:
    columns = ["r_A", *rdfs]
    table = np.column_stack([r, *(rdfs[key] for key in rdfs)])
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate RDFs for an equilibrated trajectory window.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--xyz", type=Path, default=DEFAULT_XYZ)
    parser.add_argument("--thermo", type=Path, default=DEFAULT_THERMO)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--equilibration-fraction",
        type=float,
        default=0.20,
        help="Discard this fraction of initial frames before RDF analysis.",
    )
    parser.add_argument("--cutoff-ps", type=float, default=None, help="Optionally also require frames at/after this time.")
    parser.add_argument("--rdf-frames", type=int, default=0, help="Maximum RDF frames to use after cutoff; 0 uses all.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rmax", type=float, default=5.5)
    parser.add_argument("--nbins", type=int, default=160)
    parser.add_argument("--interaction-cutoff", type=float, default=2.0)
    parser.add_argument(
        "--cores",
        "--workers",
        dest="cores",
        type=int,
        default=1,
        help="Parallel worker processes for RDF frames. Use --cores=24 on a 24-core allocation.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cores = max(1, args.cores)

    status(f"Finding trajectory in {args.run_dir}")
    xyz = args.xyz if args.xyz is not None else find_xyz(args.run_dir)
    if not xyz.is_file():
        raise FileNotFoundError(f"XYZ trajectory does not exist: {xyz}")
    if args.thermo is not None and not args.thermo.is_file():
        raise FileNotFoundError(f"Thermo file does not exist: {args.thermo}")
    status(f"Reading trajectory: {xyz}")
    frames = read(xyz, ":")
    status(f"Loaded {len(frames)} frames")
    times_ps, thermo = frame_times_ps(xyz, args.thermo, len(frames))
    if thermo is not None:
        status(f"Using thermo times from: {thermo}")
    else:
        status("No thermo file found; using frame index as time coordinate")
    rdf_idx = production_indices(
        times_ps,
        args.equilibration_fraction,
        args.cutoff_ps,
        args.rdf_frames,
        args.stride,
    )
    status(
        f"Selected {len(rdf_idx)} RDF frames from "
        f"frame {int(rdf_idx[0])} ({times_ps[int(rdf_idx[0])]:.6g} ps) to "
        f"frame {int(rdf_idx[-1])} ({times_ps[int(rdf_idx[-1])]:.6g} ps)"
    )

    rdf_frames = [frames[int(i)] for i in rdf_idx]
    validate_rmax(rdf_frames, args.rmax)
    pairs = rdf_pairs(rdf_frames)
    status(f"Computing RDFs for pairs {', '.join(f'{a}-{b}' for a, b in pairs)} using {cores} core(s)")
    r, rdfs = rdf_for_frames(rdf_frames, args.rmax, args.nbins, pairs, cores)

    rdf_path = output_dir / f"{xyz.stem}_rdf.png"
    csv_path = output_dir / f"{xyz.stem}_rdf.csv"
    status(f"Saving RDF CSV: {csv_path}")
    write_rdf_csv(csv_path, r, rdfs)
    status(f"Saving RDF plot: {rdf_path}")
    plot_rdfs(rdf_path, r, rdfs, args.interaction_cutoff)
    print(f"Input trajectory: {xyz}")
    if thermo is not None:
        print(f"Thermo file: {thermo}")
    print(f"Equilibration discarded: {100.0 * args.equilibration_fraction:.1f}% of frames")
    if args.cutoff_ps is not None:
        print(f"Additional time cutoff: {args.cutoff_ps:g} ps")
    print(f"RDF frames: {len(rdf_frames)}")
    print(f"RDF pairs: {', '.join(rdfs)}")
    print(f"Saved RDF CSV: {csv_path}")
    print(f"Saved RDF plot: {rdf_path}")


if __name__ == "__main__":
    main()
