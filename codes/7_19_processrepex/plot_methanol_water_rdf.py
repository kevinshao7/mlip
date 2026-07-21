from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell, find_mic
from ase.io import read
from scipy.io import netcdf_file


DEFAULT_REPLICA = Path(
    r"C:\Users\shaoq\Documents\Mainz\mlip\outputsfull\7_20_repex"
    r"\replica_15_lambda_1.0000_el_1.0000"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "plots"
ASEMOLEC_SOURCE = Path(__file__).resolve().parents[2] / "src" / "asemolec"
DETECTED_PYTHON = Path(r"C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe")


def load_topology(template_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    if not template_path.is_file():
        raise FileNotFoundError(f"Template PDB not found: {template_path}")

    atoms = read(template_path, index=0)
    symbols = np.array(atoms.get_chemical_symbols())
    residue_names = atoms.arrays["residuenames"]

    selections = {
        "methanol_C": np.flatnonzero((residue_names == "MOL") & (symbols == "C")),
        "methanol_O": np.flatnonzero((residue_names == "MOL") & (symbols == "O")),
        "water_O": np.flatnonzero((residue_names == "HOH") & (symbols == "O")),
        "all_O": np.flatnonzero(symbols == "O"),
    }

    missing = [name for name, indices in selections.items() if indices.size == 0]
    if missing:
        raise ValueError(f"Missing required atom selections: {', '.join(missing)}")

    return atoms.get_chemical_symbols(), selections


def iter_netcdf_frames(
    trajectory_path: Path,
    symbols: list[str],
    stride: int,
    max_frames: int | None,
):
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")

    with netcdf_file(trajectory_path, "r", mmap=False) as dataset:
        coordinates = dataset.variables["coordinates"].data
        cell_lengths = dataset.variables["cell_lengths"].data
        cell_angles = dataset.variables["cell_angles"].data
        total_frames = coordinates.shape[0]
        indices = range(0, total_frames, stride)
        if max_frames is not None:
            indices = list(indices)[:max_frames]

        for frame_index in indices:
            cellpar = np.concatenate((cell_lengths[frame_index], cell_angles[frame_index]))
            yield frame_index, Atoms(
                symbols=symbols,
                positions=np.array(coordinates[frame_index], dtype=float),
                cell=cellpar_to_cell(cellpar),
                pbc=True,
            )


def calculate_rdfs(
    replica_dir: Path,
    rmax: float,
    dr: float,
    stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    symbols, selections = load_topology(replica_dir / "minimized.pdb")

    nbins = int(np.ceil(rmax / dr))
    edges = np.linspace(0.0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 * np.pi * centers**2 * np.diff(edges)
    rdf = np.zeros(nbins)
    selection_counts = {name: int(indices.size) for name, indices in selections.items()}

    frame_count = 0
    for _frame_index, atoms in iter_netcdf_frames(
        replica_dir / "trajectory.nc", symbols, stride, max_frames
    ):
        frame_count += 1
        volume = atoms.get_volume()
        distances = atoms.get_all_distances(mic=True)
        pair_distances = distances[np.ix_(selections["methanol_C"], selections["all_O"])].ravel()
        hist, _ = np.histogram(pair_distances[(pair_distances > 0.0) & (pair_distances < rmax)], bins=edges)
        norm = selections["methanol_C"].size * (selections["all_O"].size / volume) * shell
        rdf += hist / np.maximum(norm, 1e-30)

    if frame_count == 0:
        raise ValueError("No trajectory frames were sampled.")

    rdf /= frame_count
    return centers, rdf, selection_counts, frame_count


def write_rdf_csv(centers: np.ndarray, rdf: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "carbon_oxygen_rdf_replica_15.csv"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["r_angstrom", "methanol C - all O"])
        writer.writerows(zip(centers, rdf))

    return output_path


def plot_rdfs(centers: np.ndarray, rdf: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "carbon_oxygen_rdf_replica_15.png"

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(centers, rdf, label="methanol C - all O", linewidth=1.6, color="#2166ac")

    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title("Carbon-oxygen RDF, lambda=1 replica")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate methanol-carbon to all-oxygen RDFs for the fully "
            "interacting replica-exchange replica."
        )
    )
    parser.add_argument("replica_dir", nargs="?", type=Path, default=DEFAULT_REPLICA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rmax", type=float, default=10.0)
    parser.add_argument("--dr", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.rmax <= 0:
        parser.error("--rmax must be positive")
    if args.dr <= 0:
        parser.error("--dr must be positive")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")

    centers, rdf, selection_counts, frame_count = calculate_rdfs(
        args.replica_dir,
        args.rmax,
        args.dr,
        args.stride,
        args.max_frames,
    )
    outputs = [
        write_rdf_csv(centers, rdf, args.output_dir),
        plot_rdfs(centers, rdf, args.output_dir),
    ]

    print(f"Python executable expected for this workflow: {DETECTED_PYTHON}")
    print(f"Python executable used now: {Path(sys.executable)}")
    print(f"Replica analyzed: {args.replica_dir}")
    print(f"Sampled {frame_count} frame(s) with stride {args.stride}.")
    print("Atom selections:")
    for name, count in selection_counts.items():
        print(f"  {name}: {count}")
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
