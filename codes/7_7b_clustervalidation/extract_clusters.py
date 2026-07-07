#!/usr/bin/env python3
"""Extract finite clusters from MD trajectories for ORCA validation.

This script:
1. wraps whole molecules/components with ``aseMolec.anaAtoms.wrap_molecs``,
2. selects one cluster from each saved production frame,
3. writes each cluster to its own ``.xyz`` file in vacuum,
4. generates matching ORCA ``.inp`` and Slurm ``.slurm`` files.

Default behavior is tuned for the current trajectories in
``codes/6_26_NPT_MACE/expand/MDresults``:
- 100 clusters per run,
- skip frame 0 and use frames 1..100 when available,
- prefer N-containing components as cluster centers when present,
- otherwise choose the component nearest the box center,
- include all wrapped components whose center-of-mass lies within 5.0 A of the
  chosen center component COM.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


REPO_ROOT = Path(__file__).resolve().parents[2]
ASEMOLEC_ROOT = REPO_ROOT / "aseMolec"
if str(ASEMOLEC_ROOT) not in sys.path:
    sys.path.insert(0, str(ASEMOLEC_ROOT))

from aseMolec import anaAtoms  # noqa: E402


ORCA_METHOD_LINE = "! wB97M-V def2-TZVPP TightSCF PAL8"
DEFAULT_INPUT_ROOT = REPO_ROOT / "codes" / "6_26_NPT_MACE" / "expand" / "MDresults"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "codes" / "clustervalidation" / "expand"


@dataclass(frozen=True)
class MoleculeRecord:
    mol_id: int
    atom_indices: np.ndarray
    com: np.ndarray
    symbols: tuple[str, ...]

    @property
    def has_nitrogen(self) -> bool:
        return "N" in self.symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing per-run MD trajectory subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to create cluster validation files in.",
    )
    parser.add_argument(
        "--clusters-per-run",
        type=int,
        default=100,
        help="Number of clusters to extract from each run.",
    )
    parser.add_argument(
        "--cluster-radius-angstrom",
        type=float,
        default=5.0,
        help="Include wrapped molecular components whose COM is within this radius.",
    )
    parser.add_argument(
        "--vacuum-box-angstrom",
        type=float,
        default=24.0,
        help="Cubic non-periodic box size assigned to extracted clusters.",
    )
    return parser.parse_args()


def list_runs(input_root: Path) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for run_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        traj = run_dir / f"{run_dir.name}.xyz"
        if traj.is_file():
            runs.append((run_dir.name, traj))
    if not runs:
        raise FileNotFoundError(f"No run trajectories found in {input_root}")
    return runs


def choose_frame_indices(n_frames: int, n_clusters: int) -> np.ndarray:
    if n_frames < 2:
        raise ValueError("Need at least 2 frames to skip initialization and extract clusters.")
    available = np.arange(1, n_frames, dtype=int)
    if len(available) < n_clusters:
        raise ValueError(
            f"Requested {n_clusters} clusters but only {len(available)} post-initialization "
            f"frames are available."
        )
    if len(available) == n_clusters:
        return available
    positions = np.linspace(0, len(available) - 1, num=n_clusters)
    return available[np.round(positions).astype(int)]


def build_molecule_records(atoms: Atoms, mol_coms: Atoms) -> list[MoleculeRecord]:
    mol_ids = atoms.arrays["molID"]
    records: list[MoleculeRecord] = []
    for mol_id in np.unique(mol_ids):
        atom_indices = np.flatnonzero(mol_ids == mol_id)
        mol = atoms[atom_indices]
        records.append(
            MoleculeRecord(
                mol_id=int(mol_id),
                atom_indices=atom_indices,
                com=np.array(mol_coms.positions[int(mol_id)], dtype=float),
                symbols=tuple(mol.get_chemical_symbols()),
            )
        )
    return records


def minimum_image_displacement(delta: np.ndarray, cell_lengths: np.ndarray) -> np.ndarray:
    wrapped = np.array(delta, dtype=float, copy=True)
    for axis in range(3):
        length = cell_lengths[axis]
        if length > 0.0:
            wrapped[axis] -= np.rint(wrapped[axis] / length) * length
    return wrapped


def choose_center_record(records: list[MoleculeRecord], cell_lengths: np.ndarray) -> MoleculeRecord:
    box_center = 0.5 * cell_lengths
    candidates = [record for record in records if record.has_nitrogen]
    if not candidates:
        candidates = records
    return min(
        candidates,
        key=lambda record: np.linalg.norm(
            minimum_image_displacement(record.com - box_center, cell_lengths)
        ),
    )


def extract_cluster(
    atoms: Atoms,
    radius_angstrom: float,
    vacuum_box_angstrom: float,
) -> tuple[Atoms, dict[str, object]]:
    mol_coms = anaAtoms.wrap_molecs([atoms], returnMols=True)[0]
    records = build_molecule_records(atoms, mol_coms)
    cell_lengths = np.array(atoms.cell.lengths(), dtype=float)
    center_record = choose_center_record(records, cell_lengths)

    selected_atom_indices: list[int] = []
    selected_mol_ids: list[int] = []
    for record in records:
        displacement = minimum_image_displacement(record.com - center_record.com, cell_lengths)
        if np.linalg.norm(displacement) <= radius_angstrom:
            selected_atom_indices.extend(record.atom_indices.tolist())
            selected_mol_ids.append(record.mol_id)

    selected_atom_indices = sorted(selected_atom_indices)
    cluster = atoms[selected_atom_indices]
    cluster.set_pbc(False)
    cluster.set_cell([vacuum_box_angstrom] * 3)

    center_mask = np.isin(selected_atom_indices, center_record.atom_indices)
    center_positions = cluster.positions[np.asarray(center_mask, dtype=bool)]
    center_symbols = np.array(cluster.get_chemical_symbols())[np.asarray(center_mask, dtype=bool)]
    center_masses = Atoms(symbols=center_symbols).get_masses()
    center_of_mass = np.average(center_positions, axis=0, weights=center_masses)
    cluster.positions += 0.5 * vacuum_box_angstrom - center_of_mass

    metadata = {
        "center_mol_id": center_record.mol_id,
        "center_formula": atoms[center_record.atom_indices].get_chemical_formula(),
        "selected_molecule_ids": ",".join(str(mol_id) for mol_id in selected_mol_ids),
        "cluster_radius_angstrom": radius_angstrom,
        "source_charge": int(atoms.info.get("charge", 0)),
        "source_multiplicity": int(atoms.info.get("spin", 1)),
    }
    cluster.info.update(metadata)
    return cluster, metadata


def xyz_geometry_block(atoms: Atoms) -> str:
    lines = []
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(
            f"{symbol:2s} {position[0]: .10f} {position[1]: .10f} {position[2]: .10f}"
        )
    return "\n".join(lines)


def write_orca_input(path: Path, xyz_path: Path, atoms: Atoms) -> None:
    charge = int(atoms.info.get("source_charge", atoms.info.get("charge", 0)))
    multiplicity = int(atoms.info.get("source_multiplicity", atoms.info.get("spin", 1)))
    text = (
        f"{ORCA_METHOD_LINE}\n\n"
        "%pal\n"
        "  nprocs 8\n"
        "end\n\n"
        f"* XYZ {charge} {multiplicity}\n"
        f"{xyz_geometry_block(atoms)}\n"
        "*\n"
    )
    path.write_text(text, encoding="utf-8")


def write_slurm_script(path: Path, job_name: str, inp_name: str, out_name: str) -> None:
    text = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load orca

orca {inp_name} > {out_name}
"""
    path.write_text(text, encoding="utf-8")


def write_submit_script(path: Path, slurm_paths: list[Path]) -> None:
    lines = ["#!/bin/bash", "set -e"]
    lines.extend(f"sbatch {slurm_path.name}" for slurm_path in slurm_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_windows_run_script(path: Path, inp_paths: list[Path]) -> None:
    lines = ["@echo off", "setlocal"]
    total = len(inp_paths)
    for index, inp_path in enumerate(inp_paths, start=1):
        stem = inp_path.stem
        lines.append(f"echo Running {index}/{total}: {inp_path.name}")
        lines.append(f"orca {inp_path.name} > {stem}.out")
        lines.append("if errorlevel 1 exit /b %errorlevel%")
        lines.append(f"echo Finished {index}/{total}: {stem}.out")
    lines.append("endlocal")
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def process_run(
    run_name: str,
    trajectory_path: Path,
    output_root: Path,
    clusters_per_run: int,
    cluster_radius_angstrom: float,
    vacuum_box_angstrom: float,
) -> None:
    frames = read(trajectory_path, ":")
    frame_indices = choose_frame_indices(len(frames), clusters_per_run)

    run_output_dir = output_root / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    slurm_paths: list[Path] = []
    inp_paths: list[Path] = []
    manifest_lines = [
        "# cluster_id frame_index natoms formula center_formula selected_molecule_ids"
    ]

    for cluster_number, frame_index in enumerate(frame_indices, start=1):
        atoms = frames[int(frame_index)].copy()
        cluster, metadata = extract_cluster(
            atoms=atoms,
            radius_angstrom=cluster_radius_angstrom,
            vacuum_box_angstrom=vacuum_box_angstrom,
        )
        cluster.info["source_run"] = run_name
        cluster.info["source_frame_index"] = int(frame_index)

        stem = f"{run_name}_cluster_{cluster_number:03d}"
        xyz_path = run_output_dir / f"{stem}.xyz"
        inp_path = run_output_dir / f"{stem}.inp"
        slurm_path = run_output_dir / f"{stem}.slurm"
        out_name = f"{stem}.out"

        write(xyz_path, cluster)
        write_orca_input(inp_path, xyz_path, cluster)
        write_slurm_script(slurm_path, job_name=stem, inp_name=inp_path.name, out_name=out_name)
        slurm_paths.append(slurm_path)
        inp_paths.append(inp_path)

        manifest_lines.append(
            " ".join(
                [
                    f"{cluster_number:03d}",
                    str(int(frame_index)),
                    str(len(cluster)),
                    cluster.get_chemical_formula(),
                    str(metadata["center_formula"]),
                    str(metadata["selected_molecule_ids"]),
                ]
            )
        )

    write_submit_script(run_output_dir / "submit_all.sh", slurm_paths)
    write_windows_run_script(run_output_dir / "run_all_windows.bat", inp_paths)
    (run_output_dir / "manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    runs = list_runs(args.input_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for run_name, trajectory_path in runs:
        process_run(
            run_name=run_name,
            trajectory_path=trajectory_path,
            output_root=args.output_root,
            clusters_per_run=args.clusters_per_run,
            cluster_radius_angstrom=args.cluster_radius_angstrom,
            vacuum_box_angstrom=args.vacuum_box_angstrom,
        )


if __name__ == "__main__":
    main()
