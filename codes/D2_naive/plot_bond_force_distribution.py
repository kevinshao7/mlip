#!/usr/bin/env python3
"""Plot H and O forces for O--H bonds defined by nearest oxygen.

Each H is paired with its nearest O in its frame.  The script shows H and O
force magnitudes, plus signed components along the O-to-H bond vector.  For
the components, positive is along O-to-H: away from O for H and toward H for
O.  Forces in this extxyz dataset are in eV/Angstrom (ASE's ORCA convention).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import iread


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "target_all.xyz"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "target_all_nearest_oh_force_distribution.png"


def nearest_oh_forces(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return H/O magnitudes and O--H-axis components in eV/Angstrom.

    O magnitudes include every O atom once.  O axial components include one
    value for each assigned O--H bond, so an oxygen assigned to two H atoms
    contributes two projections, each along its respective bond direction.
    """
    h_magnitudes: list[np.ndarray] = []
    h_axial_components: list[np.ndarray] = []
    o_magnitudes: list[np.ndarray] = []
    o_axial_components: list[np.ndarray] = []
    for frame_number, atoms in enumerate(iread(path, index=":"), start=1):
        symbols = np.asarray(atoms.get_chemical_symbols())
        oxygen_positions = atoms.positions[symbols == "O"]
        hydrogen_mask = symbols == "H"
        hydrogen_positions = atoms.positions[hydrogen_mask]
        forces = atoms.arrays.get("REF_forces")
        if forces is None:
            raise ValueError(f"Frame {frame_number} has no REF_forces array.")
        oxygen_forces = forces[symbols == "O"]
        hydrogen_forces = forces[hydrogen_mask]
        if len(hydrogen_positions) == 0:
            continue
        if len(oxygen_positions) == 0:
            raise ValueError(f"Frame {frame_number} contains H atoms but no O atoms.")

        displacement = hydrogen_positions[:, np.newaxis, :] - oxygen_positions[np.newaxis, :, :]
        distances = np.linalg.norm(displacement, axis=2)
        nearest_oxygen = distances.argmin(axis=1)
        bond_vectors = displacement[np.arange(len(hydrogen_positions)), nearest_oxygen]
        bond_directions = bond_vectors / np.linalg.norm(bond_vectors, axis=1)[:, np.newaxis]

        h_magnitudes.append(np.linalg.norm(hydrogen_forces, axis=1))
        h_axial_components.append(np.einsum("ij,ij->i", hydrogen_forces, bond_directions))
        o_magnitudes.append(np.linalg.norm(oxygen_forces, axis=1))
        o_axial_components.append(np.einsum(
            "ij,ij->i", oxygen_forces[nearest_oxygen], bond_directions
        ))

    if not h_magnitudes:
        raise ValueError(f"No hydrogen atoms found in {path}.")
    return (np.concatenate(h_magnitudes), np.concatenate(h_axial_components),
            np.concatenate(o_magnitudes), np.concatenate(o_axial_components))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help=f"Input extxyz file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output image (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins (default: 80)")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")

    h_magnitudes, h_axial_components, o_magnitudes, o_axial_components = nearest_oh_forces(args.input)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].hist(h_magnitudes, bins=args.bins, color="#0072B2", edgecolor="white", linewidth=0.4)
    axes[0, 0].set(title="Force magnitude on H", xlabel="|F(H)| (eV/Å)", ylabel="Count")
    axes[0, 1].hist(h_axial_components, bins=args.bins, color="#009E73", edgecolor="white", linewidth=0.4)
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(title="H force along nearest O–H bond", xlabel="F(H) · r̂(O→H) (eV/Å)", ylabel="Count")
    axes[1, 0].hist(o_magnitudes, bins=args.bins, color="#CC79A7", edgecolor="white", linewidth=0.4)
    axes[1, 0].set(title="Force magnitude on O", xlabel="|F(O)| (eV/Å)", ylabel="Count")
    axes[1, 1].hist(o_axial_components, bins=args.bins, color="#E69F00", edgecolor="white", linewidth=0.4)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(title="O force along assigned O–H bond", xlabel="F(O) · r̂(O→H) (eV/Å)", ylabel="Count")
    fig.suptitle(f"Nearest-oxygen O–H forces ({len(h_magnitudes):,} H atoms; {len(o_magnitudes):,} O atoms)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output} with {len(h_magnitudes):,} H and {len(o_magnitudes):,} O forces "
          f"(mean |F(H)|={h_magnitudes.mean():.4f}, |F(O)|={o_magnitudes.mean():.4f} eV/Å).")


if __name__ == "__main__":
    main()
