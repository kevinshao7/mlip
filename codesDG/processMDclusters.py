#!/usr/bin/env python3
"""Visualize MD trajectories and thermodynamic outputs.

The MD scripts in this repository write ASE extended XYZ trajectories and
``*_thermo.npy``/``*_thermo.txt`` arrays in ``MDresults``.  This script makes
quick inspection plots in the same ASE/Matplotlib style used in MACE tutorial
notebooks: thermodynamic traces, trajectory snapshots, composition summaries,
and simple radial distribution functions.

Outputs are written to ``MDresults/visualizations`` by default.  These are
diagnostic plots, not production statistical estimates; use ``errorsMD.py`` for
autocorrelation-aware uncertainties.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-processMDclusters")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read
from ase.visualize.plot import plot_atoms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_RESULTS_DIR = PROJECT_ROOT / "MDresults"
DEFAULT_OUTPUT_DIR = MD_RESULTS_DIR / "visualizations"
KB_EV_PER_K = 8.617333262145e-5

ELEMENT_COLORS = {
    "H": "#e8e8e8",
    "C": "#4d4d4d",
    "N": "#2f5fd0",
    "O": "#d83a34",
    "S": "#d7b32b",
}

THERMO_LABELS = {
    "temperature_K": "Temperature (K)",
    "pressure_GPa": "Pressure (GPa)",
    "potential_energy_eV_per_atom": "Potential energy (eV/atom)",
    "kinetic_energy_eV_per_atom": "Kinetic energy (eV/atom)",
    "total_energy_eV_per_atom": "Total energy (eV/atom)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MD visualization plots from MDresults trajectories."
    )
    parser.add_argument(
        "--results-dir",
        default=str(MD_RESULTS_DIR),
        help="Directory containing MD .xyz trajectories and thermo files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots should be written.",
    )
    parser.add_argument(
        "--trajectory",
        nargs="*",
        default=None,
        help="Specific trajectory .xyz files to process. Defaults to all in results-dir.",
    )
    parser.add_argument(
        "--thermo",
        nargs="*",
        default=None,
        help="Specific thermo .npy/.txt files to process. Defaults to all in results-dir.",
    )
    parser.add_argument(
        "--rdf-bin-width",
        type=float,
        default=0.05,
        help="Radial distribution bin width in Angstrom.",
    )
    parser.add_argument(
        "--rdf-rmax",
        type=float,
        default=None,
        help="Maximum RDF distance in Angstrom. Defaults to half the smallest cell length.",
    )
    parser.add_argument(
        "--rdf-max-frames",
        type=int,
        default=50,
        help="Maximum number of evenly spaced trajectory frames used for RDFs.",
    )
    parser.add_argument(
        "--snapshot-frames",
        type=int,
        default=3,
        help="Number of evenly spaced frames to show in trajectory snapshots.",
    )
    parser.add_argument(
        "--cluster-cutoff",
        type=float,
        default=3.5,
        help=(
            "Heavy-atom distance cutoff in Angstrom for grouping solute molecules "
            "into clusters."
        ),
    )
    parser.add_argument(
        "--cluster-elements",
        nargs="*",
        default=["N", "S"],
        help=(
            "Elements that mark a molecule as a solute for cluster analysis. "
            "Defaults to N and S for NH3/H2S mixtures."
        ),
    )
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")


def result_stem(path: Path) -> str:
    stem = safe_stem(path)
    return stem.removesuffix("_thermo")


def discover_inputs(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    results_dir = Path(args.results_dir)
    if args.trajectory is None:
        trajectories = sorted(results_dir.glob("*.xyz"))
    else:
        trajectories = [Path(path) for path in args.trajectory]

    if args.thermo is None:
        thermo = sorted(results_dir.glob("*_thermo.npy"))
        thermo += sorted(
            path
            for path in results_dir.glob("*_thermo.txt")
            if path.with_suffix(".npy") not in thermo
        )
    else:
        thermo = [Path(path) for path in args.thermo]

    return trajectories, thermo


def load_thermo(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected at least 3")

    columns: dict[str, np.ndarray] = {
        "time_fs": data[:, 0],
        "temperature_K": data[:, 1],
    }

    if data.shape[1] == 3:
        columns["potential_energy_eV_per_atom"] = data[:, 2]
    elif data.shape[1] == 4:
        columns["pressure_GPa"] = data[:, 2]
        columns["potential_energy_eV_per_atom"] = data[:, 3]
    elif data.shape[1] == 5:
        # Current MDclusters.py output: time, T, potential, kinetic, total.
        columns["potential_energy_eV_per_atom"] = data[:, 2]
        columns["kinetic_energy_eV_per_atom"] = data[:, 3]
        columns["total_energy_eV_per_atom"] = data[:, 4]
    else:
        columns["pressure_GPa"] = data[:, 2]
        columns["potential_energy_eV_per_atom"] = data[:, 3]
        columns["kinetic_energy_eV_per_atom"] = data[:, 4]
        columns["total_energy_eV_per_atom"] = data[:, 5]

    if "kinetic_energy_eV_per_atom" not in columns:
        columns["kinetic_energy_eV_per_atom"] = (
            1.5 * KB_EV_PER_K * columns["temperature_K"]
        )
    if "total_energy_eV_per_atom" not in columns:
        columns["total_energy_eV_per_atom"] = (
            columns["potential_energy_eV_per_atom"]
            + columns["kinetic_energy_eV_per_atom"]
        )

    return columns


def finite_xy(time_ps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(time_ps) & np.isfinite(values)
    return time_ps[mask], values[mask]


def plot_thermo(path: Path, output_dir: Path) -> Path:
    columns = load_thermo(path)
    time_ps = columns["time_fs"] / 1000.0
    series_names = [
        name
        for name in (
            "temperature_K",
            "pressure_GPa",
            "potential_energy_eV_per_atom",
            "kinetic_energy_eV_per_atom",
            "total_energy_eV_per_atom",
        )
        if name in columns
    ]

    fig, axes = plt.subplots(
        len(series_names), 1, figsize=(8.0, 1.9 * len(series_names)), sharex=True
    )
    if len(series_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, series_names):
        x, y = finite_xy(time_ps, columns[name])
        ax.plot(x, y, color="#283593", lw=1.4)
        ax.set_ylabel(THERMO_LABELS[name])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (ps)")
    fig.suptitle(path.name)
    fig.tight_layout()

    output_path = output_dir / f"{result_stem(path)}_thermo_timeseries.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_energy_temperature(path: Path, output_dir: Path) -> Path | None:
    columns = load_thermo(path)
    if "temperature_K" not in columns or "potential_energy_eV_per_atom" not in columns:
        return None
    temp = columns["temperature_K"]
    energy = columns["potential_energy_eV_per_atom"]
    mask = np.isfinite(temp) & np.isfinite(energy)
    if np.count_nonzero(mask) < 2:
        return None

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(temp[mask], energy[mask], s=24, color="#00796b", alpha=0.8)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Potential energy (eV/atom)")
    ax.set_title(path.name)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{result_stem(path)}_temperature_energy.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def read_trajectory(path: Path) -> list[Atoms]:
    frames = read(path, index=":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def frame_indices(n_frames: int, requested: int) -> list[int]:
    if n_frames <= 0:
        return []
    requested = max(1, min(requested, n_frames))
    return sorted(set(np.linspace(0, n_frames - 1, requested, dtype=int).tolist()))


def plot_snapshots(
    path: Path, frames: list[Atoms], output_dir: Path, n_snapshots: int
) -> Path | None:
    indices = frame_indices(len(frames), n_snapshots)
    if not indices:
        return None

    fig, axes = plt.subplots(1, len(indices), figsize=(4.4 * len(indices), 4.4))
    if len(indices) == 1:
        axes = [axes]

    for ax, frame_index in zip(axes, indices):
        atoms = frames[frame_index]
        plot_atoms(atoms, ax=ax, rotation=("18x,28y,0z"), radii=0.45)
        ax.set_title(f"frame {frame_index}")
        ax.set_axis_off()

    fig.suptitle(path.name)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_snapshots.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_composition(path: Path, frames: list[Atoms], output_dir: Path) -> Path:
    symbols = frames[0].get_chemical_symbols()
    unique_symbols = sorted(set(symbols), key=lambda symbol: (symbol != "H", symbol))
    counts = [symbols.count(symbol) for symbol in unique_symbols]
    colors = [ELEMENT_COLORS.get(symbol, "#888888") for symbol in unique_symbols]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(unique_symbols, counts, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xlabel("Element")
    ax.set_ylabel("Atom count")
    ax.set_title(frames[0].get_chemical_formula())
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_composition.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def cell_lengths(atoms: Atoms) -> np.ndarray:
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    return lengths


def available_pairs(symbols: Iterable[str]) -> list[tuple[str, str]]:
    present = set(symbols)
    preferred = [
        ("O", "O"),
        ("O", "H"),
        ("N", "H"),
        ("N", "O"),
        ("S", "H"),
        ("S", "O"),
        ("H", "H"),
    ]
    return [pair for pair in preferred if pair[0] in present and pair[1] in present]


def compute_pair_rdf(
    frames: list[Atoms],
    pair: tuple[str, str],
    r_max: float,
    bin_width: float,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(0.0, r_max + bin_width, bin_width)
    hist = np.zeros(len(bins) - 1, dtype=float)
    sampled_indices = frame_indices(len(frames), max_frames)
    n_sampled = 0
    n_reference_total = 0
    density_total = 0.0
    same_species = pair[0] == pair[1]

    for index in sampled_indices:
        atoms = frames[index]
        symbols = np.asarray(atoms.get_chemical_symbols())
        idx_a = np.where(symbols == pair[0])[0]
        idx_b = np.where(symbols == pair[1])[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue

        volume = atoms.get_volume()
        if not np.isfinite(volume) or volume <= 0.0:
            continue

        n_sampled += 1
        distances = atoms.get_all_distances(mic=True)
        if same_species:
            pair_distances = distances[np.ix_(idx_a, idx_a)]
            pair_distances = pair_distances[np.triu_indices(len(idx_a), k=1)]
            hist += 2.0 * np.histogram(pair_distances, bins=bins)[0]
            n_reference_total += len(idx_a)
            density_total += max(len(idx_a) - 1, 0) / volume
        else:
            pair_distances = distances[np.ix_(idx_a, idx_b)].ravel()
            hist += np.histogram(pair_distances, bins=bins)[0]
            n_reference_total += len(idx_a)
            density_total += len(idx_b) / volume

    if n_sampled == 0 or n_reference_total == 0 or density_total <= 0.0:
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, np.full_like(centers, np.nan)

    shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    mean_density = density_total / n_sampled
    expected = n_reference_total * mean_density * shell_volumes
    with np.errstate(divide="ignore", invalid="ignore"):
        rdf = hist / expected
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, rdf


def plot_rdfs(
    path: Path,
    frames: list[Atoms],
    output_dir: Path,
    r_max: float | None,
    bin_width: float,
    max_frames: int,
) -> Path | None:
    lengths = cell_lengths(frames[0])
    if len(lengths) == 0:
        return None
    r_max = float(r_max) if r_max is not None else 0.5 * float(np.min(lengths))
    if r_max <= bin_width:
        return None

    pairs = available_pairs(frames[0].get_chemical_symbols())
    if not pairs:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for pair in pairs:
        r, rdf = compute_pair_rdf(frames, pair, r_max, bin_width, max_frames)
        if np.all(~np.isfinite(rdf)):
            continue
        ax.plot(r, rdf, lw=1.5, label=f"{pair[0]}-{pair[1]}")

    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("g(r)")
    ax.set_title(path.name)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_rdf.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_bond_length_histograms(
    path: Path, frames: list[Atoms], output_dir: Path
) -> Path | None:
    pairs = available_pairs(frames[0].get_chemical_symbols())
    if not pairs:
        return None

    distances_by_pair: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    for atoms in frames:
        symbols = np.asarray(atoms.get_chemical_symbols())
        distances = atoms.get_all_distances(mic=True)
        for pair in pairs:
            idx_a = np.where(symbols == pair[0])[0]
            idx_b = np.where(symbols == pair[1])[0]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            if pair[0] == pair[1]:
                pair_distances = distances[np.ix_(idx_a, idx_a)]
                pair_distances = pair_distances[np.triu_indices(len(idx_a), k=1)]
            else:
                pair_distances = distances[np.ix_(idx_a, idx_b)].ravel()
            cutoff = (
                covalent_radii[atoms[idx_a[0]].number]
                + covalent_radii[atoms[idx_b[0]].number]
                + 0.45
            )
            distances_by_pair[pair].extend(
                pair_distances[pair_distances <= cutoff].tolist()
            )

    active_pairs = [pair for pair, values in distances_by_pair.items() if values]
    if not active_pairs:
        return None

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for pair in active_pairs:
        ax.hist(
            distances_by_pair[pair],
            bins=35,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=f"{pair[0]}-{pair[1]}",
        )
    ax.set_xlabel("Distance (Angstrom)")
    ax.set_ylabel("Density")
    ax.set_title(path.name)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_short_distance_histograms.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def covalent_molecules(atoms: Atoms) -> list[list[int]]:
    # Current systems are H2O/NH3/H2S-like hydrides. Assign each H to only its
    # nearest covalently plausible heavy atom so dense-liquid close contacts do
    # not fuse multiple molecules into one graph component.
    symbols = np.asarray(atoms.get_chemical_symbols())
    heavy_indices = [index for index, symbol in enumerate(symbols) if symbol != "H"]
    hydrogen_indices = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    molecules = {index: [index] for index in heavy_indices}
    assigned_hydrogens: set[int] = set()

    if heavy_indices and hydrogen_indices:
        distances = atoms.get_all_distances(mic=True)
        for hydrogen in hydrogen_indices:
            candidates: list[tuple[float, int]] = []
            for heavy in heavy_indices:
                cutoff = (
                    covalent_radii[atoms[hydrogen].number]
                    + covalent_radii[atoms[heavy].number]
                    + 0.35
                )
                distance = float(distances[hydrogen, heavy])
                if distance <= cutoff:
                    candidates.append((distance, heavy))
            if candidates:
                _, nearest_heavy = min(candidates, key=lambda item: item[0])
                molecules[nearest_heavy].append(hydrogen)
                assigned_hydrogens.add(hydrogen)

    components = [sorted(indices) for indices in molecules.values()]
    components.extend([[index] for index in hydrogen_indices if index not in assigned_hydrogens])
    return components


def molecule_formula(atoms: Atoms, indices: list[int]) -> str:
    symbols = [atoms[index].symbol for index in indices]
    parts = []
    for symbol in sorted(set(symbols), key=lambda item: (item != "C", item != "H", item)):
        count = symbols.count(symbol)
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def is_solute_molecule(
    atoms: Atoms, indices: list[int], cluster_elements: set[str]
) -> bool:
    symbols = {atoms[index].symbol for index in indices}
    if symbols == {"H", "O"} and len(indices) == 3:
        return False
    return bool(symbols & cluster_elements)


def solute_cluster_records(
    atoms: Atoms, frame_index: int, cluster_cutoff: float, cluster_elements: set[str]
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    molecules = covalent_molecules(atoms)
    solutes = [
        molecule
        for molecule in molecules
        if is_solute_molecule(atoms, molecule, cluster_elements)
    ]
    if not solutes:
        return [], {
            "frame": frame_index,
            "n_clusters": 0,
            "largest_cluster_size": 0,
            "mean_cluster_size": 0.0,
            "frame_energy_eV_per_atom": frame_energy_per_atom(atoms),
        }

    symbols = np.asarray(atoms.get_chemical_symbols())
    heavy_indices = [
        np.asarray([index for index in molecule if symbols[index] != "H"], dtype=int)
        for molecule in solutes
    ]
    distances = atoms.get_all_distances(mic=True)
    adjacency = [set() for _ in solutes]
    for i in range(len(solutes)):
        for j in range(i + 1, len(solutes)):
            if len(heavy_indices[i]) == 0 or len(heavy_indices[j]) == 0:
                continue
            pair_distances = distances[np.ix_(heavy_indices[i], heavy_indices[j])]
            if float(np.min(pair_distances)) <= cluster_cutoff:
                adjacency[i].add(j)
                adjacency[j].add(i)

    molecule_clusters: list[list[int]] = []
    visited: set[int] = set()
    for start in range(len(solutes)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        cluster: list[int] = []
        while stack:
            molecule_index = stack.pop()
            cluster.append(molecule_index)
            for neighbor in adjacency[molecule_index]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        molecule_clusters.append(sorted(cluster))

    atomic_energies = atomic_energy_array(atoms)
    records: list[dict[str, float | int | str]] = []
    for cluster in molecule_clusters:
        atom_indices = sorted(index for molecule_index in cluster for index in solutes[molecule_index])
        formulas = [molecule_formula(atoms, solutes[molecule_index]) for molecule_index in cluster]
        record: dict[str, float | int | str] = {
            "frame": frame_index,
            "cluster_size_molecules": len(cluster),
            "cluster_size_atoms": len(atom_indices),
            "formula": "+".join(sorted(formulas)),
        }
        if atomic_energies is not None:
            energy_sum = float(np.sum(atomic_energies[atom_indices]))
            record["atomic_energy_sum_eV"] = energy_sum
            record["atomic_energy_eV_per_molecule"] = energy_sum / len(cluster)
        records.append(record)

    sizes = [int(record["cluster_size_molecules"]) for record in records]
    frame_summary = {
        "frame": frame_index,
        "n_clusters": len(records),
        "largest_cluster_size": max(sizes),
        "mean_cluster_size": float(np.mean(sizes)),
        "frame_energy_eV_per_atom": frame_energy_per_atom(atoms),
    }
    return records, frame_summary


def atomic_energy_array(atoms: Atoms) -> np.ndarray | None:
    results = getattr(getattr(atoms, "calc", None), "results", {})
    energies = results.get("energies")
    if energies is None:
        return None
    energies = np.asarray(energies, dtype=float)
    if energies.shape != (len(atoms),):
        return None
    return energies


def frame_energy_per_atom(atoms: Atoms) -> float:
    results = getattr(getattr(atoms, "calc", None), "results", {})
    energy = results.get("energy")
    if energy is None or len(atoms) == 0:
        return float("nan")
    return float(energy) / len(atoms)


def collect_cluster_data(
    frames: list[Atoms], cluster_cutoff: float, cluster_elements: set[str]
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int]]]:
    cluster_records: list[dict[str, float | int | str]] = []
    frame_records: list[dict[str, float | int]] = []
    for frame_index, atoms in enumerate(frames):
        records, frame_summary = solute_cluster_records(
            atoms, frame_index, cluster_cutoff, cluster_elements
        )
        cluster_records.extend(records)
        frame_records.append(frame_summary)
    return cluster_records, frame_records


def plot_cluster_size_distribution(
    path: Path,
    cluster_records: list[dict[str, float | int | str]],
    output_dir: Path,
    cluster_cutoff: float,
) -> Path | None:
    if not cluster_records:
        return None
    sizes = np.asarray(
        [int(record["cluster_size_molecules"]) for record in cluster_records], dtype=int
    )
    bins = np.arange(1, int(np.max(sizes)) + 2)
    counts = np.asarray([np.count_nonzero(sizes == size) for size in bins[:-1]])

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(bins[:-1], counts, width=0.75, color="#455a64", edgecolor="#222222")
    ax.set_xlabel("Solute cluster size (molecules)")
    ax.set_ylabel("Count across saved frames")
    ax.set_title(f"{path.name}; cutoff = {cluster_cutoff:.2f} Angstrom")
    ax.set_xticks(bins[:-1])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_cluster_size_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_cluster_time_series(
    path: Path,
    frame_records: list[dict[str, float | int]],
    output_dir: Path,
) -> Path | None:
    if not frame_records:
        return None
    frames = np.asarray([int(record["frame"]) for record in frame_records], dtype=int)
    n_clusters = np.asarray([record["n_clusters"] for record in frame_records], dtype=float)
    largest = np.asarray(
        [record["largest_cluster_size"] for record in frame_records], dtype=float
    )
    mean_size = np.asarray(
        [record["mean_cluster_size"] for record in frame_records], dtype=float
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    axes[0].plot(frames, n_clusters, marker="o", color="#5d4037")
    axes[0].set_ylabel("Clusters")
    axes[1].plot(frames, largest, marker="o", color="#00695c")
    axes[1].set_ylabel("Largest size")
    axes[2].plot(frames, mean_size, marker="o", color="#6a1b9a")
    axes[2].set_ylabel("Mean size")
    axes[2].set_xlabel("Saved frame index")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(path.name)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_cluster_timeseries.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_cluster_energy_by_size(
    path: Path,
    cluster_records: list[dict[str, float | int | str]],
    output_dir: Path,
) -> Path | None:
    energy_key = "atomic_energy_eV_per_molecule"
    records = [record for record in cluster_records if energy_key in record]
    if not records:
        return None

    sizes = sorted({int(record["cluster_size_molecules"]) for record in records})
    grouped = [
        [float(record[energy_key]) for record in records if int(record["cluster_size_molecules"]) == size]
        for size in sizes
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.boxplot(grouped, positions=sizes, widths=0.6, showfliers=True)
    ax.scatter(
        [int(record["cluster_size_molecules"]) for record in records],
        [float(record[energy_key]) for record in records],
        s=18,
        color="#ef6c00",
        alpha=0.55,
        zorder=3,
    )
    ax.set_xlabel("Solute cluster size (molecules)")
    ax.set_ylabel("MACE atomic energy sum (eV/molecule)")
    ax.set_title(f"{path.name}; diagnostic per-atom energy decomposition")
    ax.set_xticks(sizes)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_cluster_energy_by_size.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_frame_energy_vs_cluster_size(
    path: Path,
    frame_records: list[dict[str, float | int]],
    output_dir: Path,
) -> Path | None:
    if not frame_records:
        return None
    largest = np.asarray(
        [record["largest_cluster_size"] for record in frame_records], dtype=float
    )
    energy = np.asarray(
        [record["frame_energy_eV_per_atom"] for record in frame_records], dtype=float
    )
    finite = np.isfinite(largest) & np.isfinite(energy)
    if np.count_nonzero(finite) < 2:
        return None

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(largest[finite], energy[finite], s=42, color="#1565c0", alpha=0.8)
    ax.set_xlabel("Largest solute cluster size (molecules)")
    ax.set_ylabel("Frame potential energy (eV/atom)")
    ax.set_title(path.name)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{safe_stem(path)}_frame_energy_vs_largest_cluster.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_cluster_figures(
    path: Path,
    frames: list[Atoms],
    output_dir: Path,
    cluster_cutoff: float,
    cluster_elements: set[str],
) -> list[Path]:
    cluster_records, frame_records = collect_cluster_data(
        frames, cluster_cutoff, cluster_elements
    )
    outputs: list[Path] = []
    for output in (
        plot_cluster_size_distribution(
            path, cluster_records, output_dir, cluster_cutoff
        ),
        plot_cluster_time_series(path, frame_records, output_dir),
        plot_cluster_energy_by_size(path, cluster_records, output_dir),
        plot_frame_energy_vs_cluster_size(path, frame_records, output_dir),
    ):
        if output is not None:
            outputs.append(output)
    return outputs


def process_trajectory(path: Path, output_dir: Path, args: argparse.Namespace) -> list[Path]:
    frames = read_trajectory(path)
    if not frames:
        raise ValueError(f"No frames found in {path}")

    outputs: list[Path] = []
    for output in (
        plot_snapshots(path, frames, output_dir, args.snapshot_frames),
        plot_composition(path, frames, output_dir),
        plot_rdfs(
            path,
            frames,
            output_dir,
            args.rdf_rmax,
            args.rdf_bin_width,
            args.rdf_max_frames,
        ),
        plot_bond_length_histograms(path, frames, output_dir),
    ):
        if output is not None:
            outputs.append(output)
    outputs.extend(
        plot_cluster_figures(
            path,
            frames,
            output_dir,
            args.cluster_cutoff,
            set(args.cluster_elements),
        )
    )
    return outputs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories, thermo_files = discover_inputs(args)
    if not trajectories and not thermo_files:
        raise SystemExit(f"No MD outputs found in {args.results_dir}")

    written: list[Path] = []
    for thermo_path in thermo_files:
        written.append(plot_thermo(thermo_path, output_dir))
        energy_plot = plot_energy_temperature(thermo_path, output_dir)
        if energy_plot is not None:
            written.append(energy_plot)

    for trajectory_path in trajectories:
        written.extend(process_trajectory(trajectory_path, output_dir, args))

    print(f"Wrote {len(written)} visualization files to {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
