#!/usr/bin/env python3
"""Plot sampled per-atom energy distributions after each head's E0 subtraction.

The ``pt_head`` uses the original labeled OMol25 replay energies and the
atomic-energy baseline embedded in the MACE-POLAR foundation checkpoint.  The
``Default`` head uses the ORCA DFT fine-tuning energies and isolated-atom DFT
E0s.  Energies in both panels are residual energies in eV per atom.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
from ase.data import atomic_numbers
from ase.io import iread

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
MACE_REPO = MLIP_DIR / "mace"
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_FOUNDATION_MODEL = MLIP_DIR / "outputsfull" / ".cache" / "mace" / "MACEPOLAR1Smodel"
DEFAULT_DFT_E0S = MLIP_DIR / "codes" / "7_7b_clustervalidation" / "atomizationenergies.txt"


def load_dft_e0s(path: Path) -> dict[int, float]:
    """Read the isolated-atom ORCA DFT references (eV) from the project CSV."""
    e0s: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().lower() == "atom":
                continue
            if len(row) != 2:
                raise ValueError(f"Expected 'element,energy' in {path}, got {row!r}")
            e0s[atomic_numbers[row[0].strip()]] = float(row[1])
    if not e0s:
        raise ValueError(f"No DFT E0s found in {path}")
    return e0s


def load_foundation_e0s(model_path: Path, head_index: int) -> dict[int, float]:
    """Read MACE-POLAR E0s from a local checkpoint without downloading it."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Foundation model is missing: {model_path}. Download/cache polar-1-s first, "
            "or provide --foundation-model."
        )
    if not MACE_REPO.is_dir():
        raise FileNotFoundError(f"Local MACE checkout is missing: {MACE_REPO}")

    sys.path.insert(0, str(MACE_REPO))
    import torch

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    z_values = model.atomic_numbers.detach().cpu().tolist()
    energies = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    energies = np.squeeze(energies)
    if energies.ndim == 2:
        if not 0 <= head_index < energies.shape[0]:
            raise ValueError(
                f"--foundation-head-index={head_index} is outside the available "
                f"range [0, {energies.shape[0] - 1}]"
            )
        energies = energies[head_index]
    if energies.ndim != 1 or len(z_values) != len(energies):
        raise ValueError(
            "Could not match foundation atomic numbers to E0s: "
            f"Z length={len(z_values)}, E0 shape={energies.shape}"
        )
    return {int(z): float(e0) for z, e0 in zip(z_values, energies, strict=True)}


def sampled_residual_energies(
    path: Path,
    e0s: dict[int, float],
    energy_key: str,
    max_configs: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Reservoir-sample ``(E - sum_i E0(Z_i)) / N_atoms`` in eV per atom."""
    if max_configs is not None and max_configs < 1:
        raise ValueError("--max-configs must be positive, or omit it to use all configurations")

    sample: list[float] = []
    total = 0
    for atoms in iread(path, index=":"):
        if energy_key not in atoms.info:
            raise KeyError(f"{path}: configuration {total} has no {energy_key!r} label")
        missing = sorted(set(int(z) for z in atoms.numbers) - set(e0s))
        if missing:
            raise KeyError(
                f"{path}: configuration {total} contains elements with no E0: {missing}"
            )
        residual = (
            float(atoms.info[energy_key]) - sum(e0s[int(z)] for z in atoms.numbers)
        ) / len(atoms)
        total += 1
        if max_configs is None or len(sample) < max_configs:
            sample.append(residual)
        else:
            replace = int(rng.integers(total))
            if replace < max_configs:
                sample[replace] = residual
    if not sample:
        raise ValueError(f"No configurations found in {path}")
    return np.asarray(sample), total


def plot_distribution(
    pt_values: np.ndarray,
    default_values: np.ndarray,
    pt_total: int,
    default_total: int,
    output: Path,
    bins: str | int,
) -> None:
    datasets = (
        ("pt_head", pt_values, pt_total, "#D55E00"),
        ("Default", default_values, default_total, "#0072B2"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, (head, values, total, color) in zip(axes, datasets, strict=True):
        axis.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.6)
        axis.axvline(np.median(values), color="black", linestyle="--", linewidth=1, label="median")
        axis.set_title(f"{head}: n={len(values):,} sampled from {total:,}")
        axis.set_xlabel(r"Energy per atom $(E - \sum_i E_0(Z_i)) / N$ (eV/atom)")
        axis.set_ylabel("Configuration count")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Training-label energy distributions after head-specific E0 subtraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def parse_bins(value: str) -> str | int:
    if value == "auto":
        return value
    bins = int(value)
    if bins < 1:
        raise argparse.ArgumentTypeError("--bins must be 'auto' or a positive integer")
    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-data", type=Path, default=DATA_DIR / "omol25_replay.extxyz")
    parser.add_argument("--default-data", type=Path, default=DATA_DIR / "target_train.xyz")
    parser.add_argument("--foundation-model", type=Path, default=DEFAULT_FOUNDATION_MODEL)
    parser.add_argument("--foundation-head-index", type=int, default=0)
    parser.add_argument("--dft-e0s", type=Path, default=DEFAULT_DFT_E0S)
    parser.add_argument("--energy-key", default="REF_energy")
    parser.add_argument(
        "--max-configs", type=int, default=5000,
        help="Uniform reservoir-sample size per dataset; use 0 to read every configuration.",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--bins", type=parse_bins, default="auto")
    parser.add_argument(
        "--output", type=Path,
        default=(
            SCRIPT_DIR
            / "runs"
            / "polar1s_mhft_orca"
            / "results"
            / "head_energy_distributions.svg"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for input_path in (args.pt_data, args.default_data, args.dft_e0s):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required input is missing: {input_path}")

    pt_e0s = load_foundation_e0s(args.foundation_model, args.foundation_head_index)
    dft_e0s = load_dft_e0s(args.dft_e0s)
    rng = np.random.default_rng(args.seed)
    max_configs = None if args.max_configs == 0 else args.max_configs
    pt_values, pt_total = sampled_residual_energies(
        args.pt_data, pt_e0s, args.energy_key, max_configs, rng
    )
    default_values, default_total = sampled_residual_energies(
        args.default_data, dft_e0s, args.energy_key, max_configs, rng
    )
    plot_distribution(pt_values, default_values, pt_total, default_total, args.output, args.bins)
    print(
        f"Wrote {args.output} (pt_head: {len(pt_values)}/{pt_total}; "
        f"Default: {len(default_values)}/{default_total}; units: eV per atom)."
    )


if __name__ == "__main__":
    main()
