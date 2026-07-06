#!/usr/bin/env python3
"""Evaluate extracted cluster energies with the same MACE-POLAR model as MD."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import mace_polar


# Match the MD scripts' CPU threading defaults unless the environment overrides them.
N_THREADS = os.environ.get("OMP_NUM_THREADS", "20")
os.environ.setdefault("OMP_NUM_THREADS", N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", N_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", N_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", N_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", N_THREADS)
os.environ.setdefault("TORCH_NUM_THREADS", N_THREADS)
os.environ.setdefault("OMP_PROC_BIND", "spread")
os.environ.setdefault("OMP_PLACES", "threads")

torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
torch.set_num_interop_threads(1)


SCRIPT_DIR = Path(__file__).resolve().parent


def run_directories(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and any(path.glob("*_cluster_*.xyz"))
    )


def cluster_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob(f"{run_dir.name}_cluster_*.xyz"))


def build_calculator():
    return mace_polar(
        model="polar-1-s",
        device=os.environ.get("MLIP_MACE_DEVICE", "cpu"),
        default_dtype="float32",
    )


def evaluate_run(run_dir: Path, calc) -> np.ndarray:
    xyz_paths = cluster_paths(run_dir)
    if not xyz_paths:
        raise FileNotFoundError(f"No cluster xyz files found in {run_dir}")

    records = []
    for index, xyz_path in enumerate(xyz_paths, start=1):
        atoms = read(xyz_path)
        atoms.calc = calc
        energy_ev = atoms.get_potential_energy()
        records.append(
            (
                index,
                xyz_path.name,
                len(atoms),
                atoms.get_chemical_formula(),
                energy_ev,
                energy_ev / len(atoms),
            )
        )

    return np.array(
        records,
        dtype=[
            ("cluster_index", np.int32),
            ("xyz_file", "U128"),
            ("natoms", np.int32),
            ("formula", "U64"),
            ("energy_eV", np.float64),
            ("energy_eV_per_atom", np.float64),
        ],
    )


def main() -> None:
    calc = build_calculator()
    combined: dict[str, np.ndarray] = {}

    for run_dir in run_directories(SCRIPT_DIR):
        result = evaluate_run(run_dir, calc)
        np.save(SCRIPT_DIR / f"{run_dir.name}_mace_energies.npy", result)
        combined[run_dir.name] = result
        print(
            run_dir.name,
            len(result),
            f"E[min,max]=({result['energy_eV'].min():.6f}, {result['energy_eV'].max():.6f}) eV",
        )

    np.save(SCRIPT_DIR / "cluster_mace_energies.npy", combined, allow_pickle=True)


if __name__ == "__main__":
    main()
