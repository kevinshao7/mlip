#!/usr/bin/env python3
"""Evaluate MACE-POLAR and MACE-OFF and make energy/force parity plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
MACE_REPO = MLIP_DIR / "mace"
DEFAULT_INPUT = SCRIPT_DIR / "representative_300.xyz"
DEFAULT_OUTPUT = MLIP_DIR / "outputsfull" / "10_6_polarrepexdiagnos" / "parity" / "test_10_frames"
DEFAULT_CACHE = MLIP_DIR / "outputsfull" / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT / ".matplotlib"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Separate directory for this exact input/settings combination.")
    parser.add_argument("--n-frames", type=int, default=10,
                        help="Deterministic, time/lambda-spread subset of the input (default: 10).")
    parser.add_argument("--polar-model", default="polar-1-m")
    parser.add_argument("--off-model", default="medium")
    parser.add_argument("--device", default=os.environ.get("MLIP_MACE_DEVICE", "cuda"))
    parser.add_argument("--polar-dtype", choices=("float32", "float64"), default="float32",
                        help="MACE-POLAR precision; float32 is its checkpoint's native precision.")
    parser.add_argument("--off-dtype", choices=("float32", "float64"), default="float64",
                        help="MACE-OFF precision; float64 is its checkpoint's native precision.")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=1, help="Spin multiplicity (default: 1, singlet).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing predictions.")
    return parser.parse_args()


def calculators(args: argparse.Namespace) -> tuple[Any, Any]:
    if MACE_REPO.is_dir():
        sys.path.insert(0, str(MACE_REPO))
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false; use --device cpu")
    from mace.calculators import mace_off, mace_polar
    polar = mace_polar(model=args.polar_model, device=args.device, default_dtype=args.polar_dtype)
    try:
        off = mace_off(model=args.off_model, device=args.device, default_dtype=args.off_dtype)
    except TypeError:
        off = mace_off(model=args.off_model, device=args.device)
    return polar, off


def selected_indices(n_available: int, n_requested: int) -> np.ndarray:
    if not 1 <= n_requested <= n_available:
        raise ValueError(f"--n-frames must be in [1, {n_available}], got {n_requested}")
    return np.floor((np.arange(n_requested) + 0.5) * n_available / n_requested).astype(int)


def minimum_mic_distance(atoms: Any) -> float:
    distances = atoms.get_all_distances(mic=True)
    return float(distances[np.triu_indices(len(atoms), k=1)].min())


def input_fingerprint(path: Path, indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(indices.astype(np.int64).tobytes())
    return digest.hexdigest()


def run_settings(args: argparse.Namespace, fingerprint: str) -> dict[str, Any]:
    return {
        "input": str(args.input.resolve()), "input_sha256_and_selection": fingerprint,
        "n_frames": args.n_frames, "polar_model": args.polar_model,
        "off_model": args.off_model, "device": args.device,
        "polar_dtype": args.polar_dtype, "off_dtype": args.off_dtype,
        "charge": args.charge, "spin_multiplicity": args.spin,
        "external_field_V_per_A": [0.0, 0.0, 0.0], "pbc_required": True,
        "minimum_allowed_mic_distance_A": 0.6,
    }


def evaluate(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    polar_path = args.output_dir / "mace_polar_predictions.xyz"
    off_path = args.output_dir / "mace_off_predictions.xyz"
    frames = list(read(args.input, index=":"))
    indices = selected_indices(len(frames), args.n_frames)
    frames = [frames[index] for index in indices]
    settings = run_settings(args, input_fingerprint(args.input, indices))
    settings_path = args.output_dir / "run_settings.json"
    if polar_path.exists() != off_path.exists() and not args.force:
        raise FileExistsError("Only one prediction file exists; use --force to regenerate a matched pair")
    if polar_path.exists() and not args.force:
        if not settings_path.is_file() or json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise FileExistsError(
                "Prediction files were made with different input or settings; choose a new --output-dir or use --force"
            )
        print("Reusing matched predictions with identical recorded settings")
        return polar_path, off_path, settings

    formula = frames[0].get_chemical_formula()
    for index, atoms in enumerate(frames):
        if not np.all(atoms.pbc):
            raise ValueError(f"Frame {index}: expected fully periodic boundary conditions, got {atoms.pbc}")
        if atoms.get_chemical_formula() != formula:
            raise ValueError(f"Frame {index}: composition differs from first selected frame")
        closest = minimum_mic_distance(atoms)
        if closest < settings["minimum_allowed_mic_distance_A"]:
            raise ValueError(
                f"Frame {index} (source replica {atoms.info.get('source_replica')}, "
                f"frame {atoms.info.get('source_frame')}): minimum-image distance {closest:.3f} A "
                "is below 0.600 A; repair or exclude this overlapping geometry before model comparison"
            )
    polar_calc, off_calc = calculators(args)
    polar_frames, off_frames = [], []
    for index, source in enumerate(frames):
        polar_atoms = source.copy()
        polar_atoms.info.update(charge=args.charge, spin=args.spin, external_field=[0.0, 0.0, 0.0])
        polar_atoms.calc = polar_calc
        polar_energy = float(polar_atoms.get_potential_energy())
        polar_forces = np.asarray(polar_atoms.get_forces(), dtype=float)
        polar_atoms.calc = None
        polar_atoms.info["model_energy_eV"] = polar_energy
        polar_atoms.arrays["model_forces_eV_per_A"] = polar_forces
        polar_frames.append(polar_atoms)

        off_atoms = source.copy()
        off_atoms.calc = off_calc
        off_energy = float(off_atoms.get_potential_energy())
        off_forces = np.asarray(off_atoms.get_forces(), dtype=float)
        off_atoms.calc = None
        off_atoms.info["model_energy_eV"] = off_energy
        off_atoms.arrays["model_forces_eV_per_A"] = off_forces
        off_frames.append(off_atoms)
        print(f"Evaluated frame {index + 1}/{len(frames)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(polar_path, polar_frames, format="extxyz")
    write(off_path, off_frames, format="extxyz")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return polar_path, off_path, settings


def parity_plot(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, output: Path) -> dict[str, float | int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difference = y - x
    metrics: dict[str, float | int] = {
        "n": int(x.size), "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
    }
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    padding = max(0.04 * (high - low), 1.0e-8)
    figure, axis = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    axis.scatter(x, y, s=14, alpha=0.55, edgecolors="none", color="#0072B2")
    axis.plot([low - padding, high + padding], [low - padding, high + padding], color="#222222", lw=1.2)
    axis.set(xlim=(low - padding, high + padding), ylim=(low - padding, high + padding),
             xlabel=xlabel, ylabel=ylabel, title=title)
    axis.set_aspect("equal", adjustable="box")
    axis.text(0.03, 0.97, f"N = {x.size:,}\nMAE = {metrics['mae']:.4g}\nRMSE = {metrics['rmse']:.4g}\nr = {metrics['pearson_r']:.5f}",
              transform=axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.9})
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return metrics


def make_outputs(args: argparse.Namespace, polar_path: Path, off_path: Path, settings: dict[str, Any]) -> None:
    polar = list(read(polar_path, index=":"))
    off = list(read(off_path, index=":"))
    if len(polar) != len(off) or any(len(a) != len(b) for a, b in zip(polar, off)):
        raise ValueError("MACE-POLAR and MACE-OFF prediction files do not describe matching frames")
    polar_e = np.array([a.info["model_energy_eV"] for a in polar], dtype=float)
    off_e = np.array([a.info["model_energy_eV"] for a in off], dtype=float)
    atom_counts = np.array([len(atoms) for atoms in polar], dtype=float)
    if not np.all(atom_counts == atom_counts[0]):
        raise ValueError("Per-atom energy parity requires a fixed atom count across selected frames")
    # Fixed composition makes a mean shift an explicit, reproducible reference alignment.
    polar_centered = polar_e - polar_e.mean()
    off_centered = off_e - off_e.mean()
    polar_f = np.concatenate([a.arrays["model_forces_eV_per_A"].reshape(-1) for a in polar])
    off_f = np.concatenate([a.arrays["model_forces_eV_per_A"].reshape(-1) for a in off])

    summary = {
        "energy_centered_eV": parity_plot(
            off_centered, polar_centered, "MACE-OFF centered energy (eV)",
            "MACE-POLAR centered energy (eV)", "Energy parity (per-model means removed)",
            args.output_dir / "energy_parity_centered.png"),
        "energy_centered_meV_per_atom": parity_plot(
            1.0e3 * off_centered / atom_counts,
            1.0e3 * polar_centered / atom_counts,
            "MACE-OFF centered energy (meV / atom)",
            "MACE-POLAR centered energy (meV / atom)",
            "Energy parity (per-model means removed; intensive units)",
            args.output_dir / "energy_parity_centered_per_atom.png"),
        "force_component_eV_per_A": parity_plot(
            off_f, polar_f, "MACE-OFF force component (eV / Å)",
            "MACE-POLAR force component (eV / Å)", "Force-component parity",
            args.output_dir / "force_parity.png"),
        "energy_reference": {
            "method": "subtract each model's sample mean; fixed composition only",
            "mace_polar_mean_eV": float(polar_e.mean()), "mace_off_mean_eV": float(off_e.mean()),
            "mean_difference_meV_per_atom": float(1.0e3 * (polar_e - off_e).mean() / atom_counts[0]),
        },
        "atoms_per_frame": int(atom_counts[0]),
        "models": {"mace_polar": args.polar_model, "mace_off": args.off_model},
        "charge": args.charge, "spin_multiplicity": args.spin,
        "settings": settings,
    }
    with (args.output_dir / "energy_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "source_replica", "source_frame", "mace_off_energy_eV",
                         "mace_polar_energy_eV", "mace_off_centered_eV", "mace_polar_centered_eV"])
        for index, (p, oe, pe, oc, pc) in enumerate(zip(polar, off_e, polar_e, off_centered, polar_centered)):
            writer.writerow([index, p.info.get("source_replica"), p.info.get("source_frame"), oe, pe, oc, pc])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote parity plots, energy table, and summary to {args.output_dir}")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Sample trajectory not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = evaluate(args)
    make_outputs(args, *paths)


if __name__ == "__main__":
    main()
