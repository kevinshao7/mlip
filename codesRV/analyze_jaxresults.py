#!/usr/bin/env python3
"""Analyze JAX-MD PDB outputs and export isolated DFT validation clusters.

This script is intentionally conservative: it reads the generated PDB snapshots
as molecular records, uses periodic distances only for analysis, and writes
cluster exports as non-periodic XYZ files. The exported XYZ files should be
treated as geometry starting points for DFT validation, not as equilibrated gas
phase structures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ase import Atoms
    from ase.io import write
except ImportError as exc:  # pragma: no cover - dependency check for CLI users
    raise SystemExit(
        "ASE is required for XYZ cluster export. Install ase in the project environment."
    ) from exc


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "jaxresults"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "analysis"
ANGSTROM3_TO_CM3 = 1.0e-24
AMU_TO_G = 1.66053906660e-24
DEFAULT_TARGET_PRESSURE_BAR = 1.0

ATOMIC_MASSES = {
    "H": 1.00794,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "S": 32.065,
}

ELEMENT_COLORS = {
    "H": "#d9d9d9",
    "C": "#333333",
    "N": "#2f5fd0",
    "O": "#d83a34",
    "S": "#d2a72c",
}


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    name: str
    resname: str
    chain: str
    resid: int
    element: str
    position: np.ndarray


@dataclass
class Molecule:
    index: int
    resname: str
    resid: int
    atom_indices: list[int]


@dataclass
class PDBSystem:
    name: str
    path: Path
    atoms: list[AtomRecord]
    molecules: list[Molecule]
    cell_lengths: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze jaxresults PDB snapshots and water NPT thermo output."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing pt_output and run_water_npt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plots, CSV files, and isolated cluster exports.",
    )
    parser.add_argument(
        "--molecular-cutoff",
        type=float,
        default=3.5,
        help="Heavy-atom distance cutoff in Angstrom for molecule graph clusters.",
    )
    parser.add_argument(
        "--hydration-cutoff",
        type=float,
        default=3.5,
        help="Heavy-atom solute-water cutoff in Angstrom for hydration statistics.",
    )
    parser.add_argument(
        "--rdf-bin-width",
        type=float,
        default=0.05,
        help="RDF bin width in Angstrom.",
    )
    parser.add_argument(
        "--rdf-rmax",
        type=float,
        default=None,
        help="Maximum RDF radius in Angstrom. Defaults to half the smallest box length.",
    )
    parser.add_argument(
        "--export-water-counts",
        type=int,
        nargs="*",
        default=[0, 1, 4, 8, 12],
        help="Numbers of nearest waters to include around the solute for DFT clusters.",
    )
    parser.add_argument(
        "--water-reference-counts",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8],
        help="Numbers of nearest waters to export from the pure-water snapshot.",
    )
    return parser.parse_args()


def infer_element(line: str, atom_name: str) -> str:
    raw = line[76:78].strip() if len(line) >= 78 else ""
    if raw:
        return raw[0].upper() + raw[1:].lower()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        raise ValueError(f"Could not infer element from atom name {atom_name!r}")
    return letters[0].upper() + (letters[1].lower() if len(letters) > 1 else "")


def parse_pdb(path: Path, name: str) -> PDBSystem:
    atoms: list[AtomRecord] = []
    molecules: list[Molecule] = []
    current_indices: list[int] = []
    current_key: tuple[str, int] | None = None
    cell_lengths = np.array([math.nan, math.nan, math.nan], dtype=float)

    def flush_molecule() -> None:
        nonlocal current_indices, current_key
        if not current_indices or current_key is None:
            current_indices = []
            current_key = None
            return
        resname, resid = current_key
        molecules.append(
            Molecule(
                index=len(molecules),
                resname=resname,
                resid=resid,
                atom_indices=current_indices,
            )
        )
        current_indices = []
        current_key = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "CRYST1":
                cell_lengths = np.array(
                    [float(line[6:15]), float(line[15:24]), float(line[24:33])],
                    dtype=float,
                )
                continue
            if record == "TER":
                flush_molecule()
                continue
            if record not in {"ATOM", "HETATM"}:
                continue

            serial = int(line[6:11])
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resid = int(line[22:26])
            position = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
            element = infer_element(line, atom_name)
            atom_index = len(atoms)
            atoms.append(
                AtomRecord(
                    serial=serial,
                    name=atom_name,
                    resname=resname,
                    chain=chain,
                    resid=resid,
                    element=element,
                    position=position,
                )
            )

            key = (resname, resid)
            if current_key is not None and key != current_key:
                flush_molecule()
            current_key = key
            current_indices.append(atom_index)

    flush_molecule()
    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    if not np.all(np.isfinite(cell_lengths)) or np.any(cell_lengths <= 0.0):
        raise ValueError(f"No valid CRYST1 box found in {path}")
    return PDBSystem(name=name, path=path, atoms=atoms, molecules=molecules, cell_lengths=cell_lengths)


def positions(system: PDBSystem, atom_indices: Iterable[int] | None = None) -> np.ndarray:
    indices = list(range(len(system.atoms))) if atom_indices is None else list(atom_indices)
    return np.array([system.atoms[index].position for index in indices], dtype=float)


def symbols(system: PDBSystem, atom_indices: Iterable[int] | None = None) -> list[str]:
    indices = list(range(len(system.atoms))) if atom_indices is None else list(atom_indices)
    return [system.atoms[index].element for index in indices]


def minimum_image_delta(delta: np.ndarray, cell_lengths: np.ndarray) -> np.ndarray:
    return delta - cell_lengths * np.round(delta / cell_lengths)


def pairwise_distances(
    a: np.ndarray, b: np.ndarray, cell_lengths: np.ndarray, same_set: bool = False
) -> np.ndarray:
    delta = a[:, None, :] - b[None, :, :]
    delta = minimum_image_delta(delta, cell_lengths)
    distances = np.linalg.norm(delta, axis=2)
    if same_set:
        np.fill_diagonal(distances, np.inf)
    return distances


def heavy_atom_indices(system: PDBSystem, molecule: Molecule) -> list[int]:
    return [index for index in molecule.atom_indices if system.atoms[index].element != "H"]


def molecule_min_distance(system: PDBSystem, mol_a: Molecule, mol_b: Molecule) -> float:
    heavy_a = heavy_atom_indices(system, mol_a)
    heavy_b = heavy_atom_indices(system, mol_b)
    if not heavy_a or not heavy_b:
        return math.inf
    dmat = pairwise_distances(positions(system, heavy_a), positions(system, heavy_b), system.cell_lengths)
    return float(np.min(dmat))


def molecular_clusters(system: PDBSystem, cutoff: float) -> list[list[int]]:
    adjacency = [set() for _ in system.molecules]
    for i, mol_i in enumerate(system.molecules):
        for j in range(i + 1, len(system.molecules)):
            if molecule_min_distance(system, mol_i, system.molecules[j]) <= cutoff:
                adjacency[i].add(j)
                adjacency[j].add(i)

    clusters: list[list[int]] = []
    visited: set[int] = set()
    for start in range(len(system.molecules)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        cluster: list[int] = []
        while stack:
            current = stack.pop()
            cluster.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        clusters.append(sorted(cluster))
    return clusters


def molecule_formula(system: PDBSystem, molecule: Molecule) -> str:
    counts: dict[str, int] = {}
    for atom_index in molecule.atom_indices:
        element = system.atoms[atom_index].element
        counts[element] = counts.get(element, 0) + 1
    ordered = sorted(counts, key=lambda item: (item != "C", item != "H", item))
    return "".join(element if counts[element] == 1 else f"{element}{counts[element]}" for element in ordered)


def is_water(system: PDBSystem, molecule: Molecule) -> bool:
    return molecule.resname == "HOH" or molecule_formula(system, molecule) == "H2O"


def solute_molecules(system: PDBSystem) -> list[Molecule]:
    return [molecule for molecule in system.molecules if not is_water(system, molecule)]


def nearest_waters_to_atoms(system: PDBSystem, atom_indices: list[int]) -> list[tuple[float, Molecule]]:
    reference_positions = positions(system, atom_indices)
    waters = [molecule for molecule in system.molecules if is_water(system, molecule)]
    result: list[tuple[float, Molecule]] = []
    for water in waters:
        heavy = heavy_atom_indices(system, water)
        if not heavy:
            continue
        dmat = pairwise_distances(reference_positions, positions(system, heavy), system.cell_lengths)
        result.append((float(np.min(dmat)), water))
    return sorted(result, key=lambda item: item[0])


def unwrap_cluster_positions(system: PDBSystem, atom_indices: list[int], anchor_index: int) -> np.ndarray:
    raw = positions(system, atom_indices)
    anchor = system.atoms[anchor_index].position
    unwrapped = []
    for atom_index in atom_indices:
        delta = system.atoms[atom_index].position - anchor
        unwrapped.append(anchor + minimum_image_delta(delta, system.cell_lengths))
    coords = np.array(unwrapped, dtype=float)
    coords -= coords.mean(axis=0)
    return coords


def write_cluster_xyz(
    system: PDBSystem,
    molecules: list[Molecule],
    output_path: Path,
    comment: str,
) -> None:
    atom_indices = [atom_index for molecule in molecules for atom_index in molecule.atom_indices]
    anchor_index = atom_indices[0]
    atoms = Atoms(
        symbols=symbols(system, atom_indices),
        positions=unwrap_cluster_positions(system, atom_indices, anchor_index),
        pbc=False,
    )
    atoms.info["comment"] = comment
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, atoms, format="xyz")


def write_composition_csv(system: PDBSystem, output_path: Path) -> None:
    element_counts: dict[str, int] = {}
    molecule_counts: dict[str, int] = {}
    for atom in system.atoms:
        element_counts[atom.element] = element_counts.get(atom.element, 0) + 1
    for molecule in system.molecules:
        formula = molecule_formula(system, molecule)
        molecule_counts[formula] = molecule_counts.get(formula, 0) + 1

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "label", "count"])
        for element, count in sorted(element_counts.items()):
            writer.writerow(["element", element, count])
        for formula, count in sorted(molecule_counts.items()):
            writer.writerow(["molecule_formula", formula, count])


def density_g_cm3(system: PDBSystem) -> float:
    mass_amu = sum(ATOMIC_MASSES.get(atom.element, 0.0) for atom in system.atoms)
    volume_a3 = float(np.prod(system.cell_lengths))
    return mass_amu * AMU_TO_G / (volume_a3 * ANGSTROM3_TO_CM3)


def write_cluster_stats(system: PDBSystem, output_path: Path, cutoff: float) -> dict[str, float]:
    clusters = molecular_clusters(system, cutoff)
    sizes = np.array([len(cluster) for cluster in clusters], dtype=int)
    solute_count = len(solute_molecules(system))
    summary = {
        "n_atoms": float(len(system.atoms)),
        "n_molecules": float(len(system.molecules)),
        "n_solutes": float(solute_count),
        "box_a": float(system.cell_lengths[0]),
        "box_b": float(system.cell_lengths[1]),
        "box_c": float(system.cell_lengths[2]),
        "density_g_cm3_from_pdb": density_g_cm3(system),
        "cluster_cutoff_A": float(cutoff),
        "n_molecular_clusters": float(len(clusters)),
        "largest_molecular_cluster": float(np.max(sizes)),
        "mean_molecular_cluster_size": float(np.mean(sizes)),
    }
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    return summary


def plot_composition(system: PDBSystem, output_path: Path) -> None:
    counts: dict[str, int] = {}
    for atom in system.atoms:
        counts[atom.element] = counts.get(atom.element, 0) + 1
    elements = sorted(counts, key=lambda element: (element != "H", element))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar(
        elements,
        [counts[element] for element in elements],
        color=[ELEMENT_COLORS.get(element, "#777777") for element in elements],
        edgecolor="#222222",
        linewidth=0.7,
    )
    ax.set_xlabel("Element")
    ax.set_ylabel("Atom count")
    ax.set_title(system.name)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_rdf(system: PDBSystem, output_path: Path, bin_width: float, rmax: float | None) -> None:
    all_symbols = np.array(symbols(system))
    pos = positions(system)
    rmax = float(rmax) if rmax is not None else 0.5 * float(np.min(system.cell_lengths))
    bins = np.arange(0.0, rmax + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    volume = float(np.prod(system.cell_lengths))
    pairs = [("O", "O"), ("C", "O"), ("O", "H"), ("C", "H"), ("H", "H")]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for elem_a, elem_b in pairs:
        idx_a = np.where(all_symbols == elem_a)[0]
        idx_b = np.where(all_symbols == elem_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        same = elem_a == elem_b
        dmat = pairwise_distances(pos[idx_a], pos[idx_b], system.cell_lengths, same_set=same)
        distances = dmat[np.isfinite(dmat)].ravel()
        hist = np.histogram(distances, bins=bins)[0].astype(float)
        if same:
            n_ref = len(idx_a)
            rho = max(len(idx_a) - 1, 0) / volume
        else:
            n_ref = len(idx_a)
            rho = len(idx_b) / volume
        shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        expected = n_ref * rho * shell_volumes
        with np.errstate(divide="ignore", invalid="ignore"):
            rdf = hist / expected
        ax.plot(centers, rdf, lw=1.4, label=f"{elem_a}-{elem_b}")
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("g(r)")
    ax.set_title(system.name)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_neighbor_distribution(system: PDBSystem, output_path: Path, cutoff: float) -> None:
    nearest_counts = []
    for molecule in system.molecules:
        if not is_water(system, molecule):
            continue
        heavy = heavy_atom_indices(system, molecule)
        if not heavy:
            continue
        count = 0
        for other in system.molecules:
            if other.index == molecule.index or not is_water(system, other):
                continue
            if molecule_min_distance(system, molecule, other) <= cutoff:
                count += 1
        nearest_counts.append(count)
    if not nearest_counts:
        return
    max_count = max(nearest_counts)
    bins = np.arange(-0.5, max_count + 1.5, 1.0)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(nearest_counts, bins=bins, color="#546e7a", edgecolor="#222222")
    ax.set_xlabel(f"Water neighbors within {cutoff:.2f} Angstrom")
    ax.set_ylabel("Water molecule count")
    ax.set_title(system.name)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def split_thermo_blocks(data: np.ndarray) -> list[np.ndarray]:
    """Split appended thermo logs at step/time resets.

    The JAX output file may be appended across repeated runs. Plotting it as one
    continuous line creates artificial diagonal connections across resets.
    """
    if len(data) == 0:
        return []
    breaks = [0]
    for index in range(1, len(data)):
        if data["time"][index] <= data["time"][index - 1] or data["step"][index] <= data["step"][index - 1]:
            breaks.append(index)
    breaks.append(len(data))
    return [data[start:end] for start, end in zip(breaks[:-1], breaks[1:]) if end > start]


def plot_thermo_timeseries(thermo_path: Path, output_path: Path) -> list[dict[str, float]]:
    data = np.genfromtxt(thermo_path, delimiter=",", names=True, comments="#")
    if data.ndim == 0:
        data = np.array([data])
    blocks = split_thermo_blocks(data)
    fields = ["temperature", "potential_energy", "total_energy", "pressure", "density", "volume"]
    labels = {
        "temperature": "Temperature (K)",
        "potential_energy": "Potential energy (eV)",
        "total_energy": "Total energy (eV)",
        "pressure": "Logged instantaneous pressure (raw units)",
        "density": "Density (g/cm3)",
        "volume": "Volume (Angstrom3)",
    }
    fig, axes = plt.subplots(len(fields), 1, figsize=(8.0, 9.6), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, field in zip(axes, fields):
        for block_index, block in enumerate(blocks, start=1):
            time_ps = block["time"] / 1000.0
            ax.plot(
                time_ps,
                block[field],
                lw=1.2,
                color=colors[(block_index - 1) % len(colors)],
                label=f"block {block_index}" if field == fields[0] else None,
            )
        if field == "pressure":
            ax.axhline(
                DEFAULT_TARGET_PRESSURE_BAR,
                color="#7f7f7f",
                lw=0.9,
                ls="--",
            )
            ax.text(
                0.99,
                0.86,
                "configured target = 1 bar; log units unverified",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )
        ax.set_ylabel(labels[field])
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncols=min(len(blocks), 4))
    axes[-1].set_xlabel("Time (ps)")
    fig.suptitle(f"{thermo_path.parent.name}: appended thermo blocks split at time resets")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)

    zoom_output_path = output_path.with_name(f"{output_path.stem}_robust_zoom{output_path.suffix}")
    fig, axes = plt.subplots(len(fields), 1, figsize=(8.0, 9.6), sharex=True)
    for ax, field in zip(axes, fields):
        finite_values = []
        for block_index, block in enumerate(blocks, start=1):
            time_ps = block["time"] / 1000.0
            values = np.asarray(block[field], dtype=float)
            finite_values.extend(values[np.isfinite(values)].tolist())
            ax.plot(
                time_ps,
                values,
                lw=1.2,
                color=colors[(block_index - 1) % len(colors)],
                label=f"block {block_index}" if field == fields[0] else None,
            )
        if field == "pressure":
            ax.axhline(
                DEFAULT_TARGET_PRESSURE_BAR,
                color="#7f7f7f",
                lw=0.9,
                ls="--",
            )
            ax.text(
                0.99,
                0.86,
                "configured target = 1 bar; log units unverified",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )
        if finite_values:
            lo, hi = np.percentile(finite_values, [1.0, 95.0])
            if hi > lo:
                pad = 0.08 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(labels[field])
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncols=min(len(blocks), 4))
    axes[-1].set_xlabel("Time (ps)")
    fig.suptitle(f"{thermo_path.parent.name}: robust zoom, outliers clipped by axis limits")
    fig.tight_layout()
    fig.savefig(zoom_output_path, dpi=250)
    plt.close(fig)

    summaries: list[dict[str, float]] = []
    for block_index, block in enumerate(blocks, start=1):
        half = len(block) // 2
        density_tail = np.asarray(block["density"][half:], dtype=float)
        temp_tail = np.asarray(block["temperature"][half:], dtype=float)
        summaries.append(
            {
                "block": float(block_index),
                "start_row": float(sum(len(item) for item in blocks[: block_index - 1])),
                "n_rows": float(len(block)),
                "start_step": float(block["step"][0]),
                "final_step": float(block["step"][-1]),
                "final_time_ps": float(block["time"][-1] / 1000.0),
                "density_mean_second_half_g_cm3": float(np.mean(density_tail)),
                "density_std_second_half_g_cm3": float(np.std(density_tail, ddof=1)) if len(density_tail) > 1 else 0.0,
                "temperature_mean_second_half_K": float(np.mean(temp_tail)),
                "temperature_std_second_half_K": float(np.std(temp_tail, ddof=1)) if len(temp_tail) > 1 else 0.0,
                "final_density_g_cm3": float(block["density"][-1]),
                "final_temperature_K": float(block["temperature"][-1]),
            }
        )
    return summaries


def write_thermo_summary(summaries: list[dict[str, float]], output_path: Path) -> None:
    if not summaries:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summaries[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def export_pt_clusters(system: PDBSystem, output_dir: Path, water_counts: list[int]) -> list[Path]:
    solutes = solute_molecules(system)
    if not solutes:
        return []
    solute = solutes[0]
    solute_heavy = heavy_atom_indices(system, solute)
    nearest = nearest_waters_to_atoms(system, solute_heavy)
    written: list[Path] = []
    for count in sorted(set(water_counts)):
        waters = [water for _, water in nearest[:count]]
        output_path = output_dir / f"{system.name}_solute_plus_{count:02d}waters.xyz"
        write_cluster_xyz(
            system,
            [solute] + waters,
            output_path,
            f"{system.name}: solute {molecule_formula(system, solute)} plus {count} nearest waters",
        )
        written.append(output_path)
    return written


def export_water_clusters(system: PDBSystem, output_dir: Path, water_counts: list[int]) -> list[Path]:
    waters = [molecule for molecule in system.molecules if is_water(system, molecule)]
    if not waters:
        return []
    center = waters[len(waters) // 2]
    center_heavy = heavy_atom_indices(system, center)
    nearest = nearest_waters_to_atoms(system, center_heavy)
    nearest = [(dist, water) for dist, water in nearest if water.index != center.index]
    written: list[Path] = []
    for count in sorted(set(water_counts)):
        if count <= 0:
            continue
        selected = [center] + [water for _, water in nearest[: max(0, count - 1)]]
        output_path = output_dir / f"{system.name}_{count:02d}water_cluster.xyz"
        write_cluster_xyz(
            system,
            selected,
            output_path,
            f"{system.name}: {count} nearest-water isolated cluster from PDB snapshot",
        )
        written.append(output_path)
    return written


def analyze_system(system: PDBSystem, output_dir: Path, args: argparse.Namespace) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    composition_csv = output_dir / f"{system.name}_composition.csv"
    write_composition_csv(system, composition_csv)
    written.append(composition_csv)

    stats_csv = output_dir / f"{system.name}_cluster_summary.csv"
    write_cluster_stats(system, stats_csv, args.molecular_cutoff)
    written.append(stats_csv)

    composition_png = output_dir / f"{system.name}_composition.png"
    plot_composition(system, composition_png)
    written.append(composition_png)

    rdf_png = output_dir / f"{system.name}_rdf.png"
    plot_rdf(system, rdf_png, args.rdf_bin_width, args.rdf_rmax)
    written.append(rdf_png)

    neighbor_png = output_dir / f"{system.name}_water_neighbor_distribution.png"
    plot_neighbor_distribution(system, neighbor_png, args.hydration_cutoff)
    if neighbor_png.exists():
        written.append(neighbor_png)

    clusters_dir = output_dir / "clusters"
    written.extend(export_pt_clusters(system, clusters_dir, args.export_water_counts))
    written.extend(export_water_clusters(system, clusters_dir, args.water_reference_counts))
    return written


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_inputs = [
        (results_root / "pt_output" / "CO_O.pdb", "pt_output_CO_O"),
        (results_root / "run_water_npt" / "O_O.pdb", "run_water_npt_O_O"),
    ]
    written: list[Path] = []
    for pdb_path, name in pdb_inputs:
        if not pdb_path.is_file():
            print(f"Skipping missing PDB: {pdb_path}")
            continue
        system = parse_pdb(pdb_path, name)
        written.extend(analyze_system(system, output_dir / name, args))

    thermo_path = results_root / "run_water_npt" / "thermo.csv"
    if thermo_path.is_file():
        thermo_plot = output_dir / "run_water_npt_thermo_timeseries.png"
        summary = plot_thermo_timeseries(thermo_path, thermo_plot)
        summary_csv = output_dir / "run_water_npt_thermo_summary.csv"
        write_thermo_summary(summary, summary_csv)
        written.extend(
            [
                thermo_plot,
                thermo_plot.with_name(f"{thermo_plot.stem}_robust_zoom{thermo_plot.suffix}"),
                summary_csv,
            ]
        )
    else:
        print(f"Skipping missing thermo CSV: {thermo_path}")

    print(f"Wrote {len(written)} analysis files under {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
