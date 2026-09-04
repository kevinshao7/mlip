#!/usr/bin/env python3
"""Plot nearest-oxygen O--H bond lengths from an extxyz trajectory.

Each hydrogen is assigned to the oxygen with the smallest distance in the
same structure.  This intentionally does not require a distance cutoff, so
there is exactly one reported length per hydrogen atom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import iread


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "target_all.xyz"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "target_all_nearest_oh_lengths.png"


def nearest_oxygen_hydrogen_lengths(path: Path) -> np.ndarray:
    """Return one nearest-O distance (Angstrom) for every H in ``path``."""
    lengths: list[np.ndarray] = []
    for frame_number, atoms in enumerate(iread(path, index=":"), start=1):
        symbols = np.asarray(atoms.get_chemical_symbols())
        oxygen_positions = atoms.positions[symbols == "O"]
        hydrogen_positions = atoms.positions[symbols == "H"]
        if len(hydrogen_positions) == 0:
            continue
        if len(oxygen_positions) == 0:
            raise ValueError(f"Frame {frame_number} contains H atoms but no O atoms.")

        # Shape: (number of H, number of O); no PBC is used because this
        # dataset declares pbc=\"F F F\".
        distances = np.linalg.norm(
            hydrogen_positions[:, np.newaxis, :] - oxygen_positions[np.newaxis, :, :], axis=2
        )
        lengths.append(distances.min(axis=1))

    if not lengths:
        raise ValueError(f"No hydrogen atoms found in {path}.")
    return np.concatenate(lengths)


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

    lengths = nearest_oxygen_hydrogen_lengths(args.input)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(lengths, bins=args.bins, color="#0072B2", edgecolor="white", linewidth=0.4)
    ax.axvline(lengths.mean(), color="#D55E00", linestyle="--", linewidth=1.5,
               label=f"mean = {lengths.mean():.3f} Å")
    ax.set(xlabel="Nearest O–H distance (Å)", ylabel="Count",
           title=f"Nearest-oxygen O–H bond lengths ({len(lengths):,} H atoms)")
    ax.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output} with {len(lengths):,} bond lengths "
          f"(mean={lengths.mean():.4f} Å, min={lengths.min():.4f} Å, max={lengths.max():.4f} Å).")


if __name__ == "__main__":
    main()
