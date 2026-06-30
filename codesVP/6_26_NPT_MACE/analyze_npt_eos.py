#!/usr/bin/env python3
"""Analyze NPT trajectories for pressure, density, and temperature.

The default input is the r02_cold_w water run in expand/MDresults.  The script
reads extended XYZ trajectories with ASE, reconstructs thermodynamic time series
from the saved cell, momenta, and stress, and writes summary tables plus plots.

Pressure is computed from ASE stress with the ideal-gas kinetic contribution
included, matching the convention used in the NPT production scripts:

    pressure_GPa = -mean(stress_xx, stress_yy, stress_zz) / ase.units.GPa

Density is reported in g/cm^3 from the total atomic mass and cell volume.
Temperature is computed from the saved momenta using ASE's kinetic temperature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.io import read


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "expand" / "MDresults" / "r06_cold_w"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "npt_analysis" / "r06_cold_w"

AMU_TO_G = 1.66053906660e-24
ANGSTROM3_TO_CM3 = 1.0e-24


@dataclass(frozen=True)
class Estimate:
    mean: float
    std: float
    naive_se: float
    block_se: float
    block_size: int
    n_blocks: int
    tau_int_frames: float
    tau_se: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze NPT extended XYZ trajectories and plot pressure, density, "
            "and temperature for equation-of-state checks."
        )
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        default=DEFAULT_RUN_DIR / "r06_cold_w.xyz",
        help="Trajectory to analyze. Defaults to expand/MDresults/r06_cold_w/r06_cold_w.xyz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and PNG outputs.",
    )
    parser.add_argument(
        "--discard",
        type=int,
        default=0,
        help="Number of saved frames to discard as equilibration before statistics.",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=0.1,
        help="MD timestep in fs used by the production run.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Number of MD steps between saved XYZ frames.",
    )
    return parser.parse_args()


def run_id_from_path(xyz_path: Path) -> str:
    return xyz_path.stem


def generated_script_path(xyz_path: Path) -> Path | None:
    """Return the generated NPT script corresponding to a trajectory, if present."""
    run_id = run_id_from_path(xyz_path)
    candidate = SCRIPT_DIR / "expand" / f"npt_{run_id}.py"
    return candidate if candidate.exists() else None


def parse_run_settings(script_path: Path | None) -> dict[str, float | str]:
    """Extract target T/P/rho from a generated script without executing it."""
    if script_path is None:
        return {}

    text = script_path.read_text(encoding="utf-8")
    patterns = {
        "target_density_g_cm3": r"^densitygcm3\s*=\s*([0-9.eE+-]+)",
        "target_pressure_GPa": r"^pressuregpa\s*=\s*([0-9.eE+-]+)",
        "target_temperature_K": r"^\s*temp\s*=\s*([0-9.eE+-]+),",
    }
    settings: dict[str, float | str] = {"generated_script": str(script_path)}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            settings[key] = float(match.group(1))
    return settings


def density_g_cm3(atoms) -> float:
    mass_g = float(np.sum(atoms.get_masses())) * AMU_TO_G
    volume_cm3 = float(atoms.get_volume()) * ANGSTROM3_TO_CM3
    return mass_g / volume_cm3


def pressure_gpa(atoms) -> float:
    """Hydrostatic pressure in GPa, including ideal-gas kinetic stress."""
    stress = atoms.get_stress(include_ideal_gas=True)
    return float(-np.mean(stress[:3]) / units.GPa)


def frame_table(xyz_path: Path, timestep_fs: float, save_interval: int) -> list[dict[str, float]]:
    frames = read(xyz_path, ":")
    rows: list[dict[str, float]] = []
    saved_dt_fs = timestep_fs * save_interval

    for frame_index, atoms in enumerate(frames):
        rows.append(
            {
                "frame": float(frame_index),
                "time_fs": frame_index * saved_dt_fs,
                "temperature_K": float(atoms.get_temperature()),
                "pressure_GPa": pressure_gpa(atoms),
                "density_g_cm3": density_g_cm3(atoms),
                "volume_A3": float(atoms.get_volume()),
                "potential_energy_eV_per_atom": float(atoms.get_potential_energy()) / len(atoms),
                "kinetic_energy_eV_per_atom": float(atoms.get_kinetic_energy()) / len(atoms),
                "n_atoms": float(len(atoms)),
            }
        )
    return rows


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation function using an FFT convolution."""
    centered = values - np.mean(values)
    n = len(centered)
    if n < 2 or np.allclose(centered, 0.0):
        return np.ones(n)

    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_len)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_len)[:n]
    acf /= np.arange(n, 0, -1)
    acf /= acf[0]
    return np.real(acf)


def tau_int_initial_positive(values: np.ndarray) -> float:
    """Estimate integrated autocorrelation time in saved-frame units.

    The sum is truncated at the first non-positive ACF value.  This is a simple,
    conservative diagnostic for short MD runs; block averaging below is the main
    uncertainty estimate reported for production-length trajectories.
    """
    acf = autocorrelation_fft(values)
    tau = 0.5
    for rho in acf[1:]:
        if rho <= 0.0:
            break
        tau += float(rho)
    return max(tau, 0.5)


def block_standard_error(values: np.ndarray) -> tuple[float, int, int]:
    """Return block-averaged SE using the largest block with at least 4 blocks."""
    n = len(values)
    if n < 8:
        return float("nan"), 1, n

    chosen_se = float("nan")
    chosen_block = 1
    chosen_n_blocks = n
    block_size = 1
    while block_size <= n // 4:
        n_blocks = n // block_size
        trimmed = values[: n_blocks * block_size]
        block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
        if n_blocks > 1:
            chosen_se = float(np.std(block_means, ddof=1) / math.sqrt(n_blocks))
            chosen_block = block_size
            chosen_n_blocks = n_blocks
        block_size *= 2
    return chosen_se, chosen_block, chosen_n_blocks


def estimate(values: Iterable[float]) -> Estimate:
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot estimate statistics for an empty array.")

    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    naive_se = std / math.sqrt(n) if n > 1 else float("nan")
    block_se, block_size, n_blocks = block_standard_error(arr)
    tau = tau_int_initial_positive(arr) if n > 1 else float("nan")
    tau_se = std * math.sqrt(2.0 * tau / n) if n > 1 and math.isfinite(tau) else float("nan")
    return Estimate(
        mean=float(np.mean(arr)),
        std=std,
        naive_se=naive_se,
        block_se=block_se,
        block_size=block_size,
        n_blocks=n_blocks,
        tau_int_frames=tau,
        tau_se=tau_se,
    )


def write_time_series_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, run_id: str, n_used: int, settings: dict, estimates: dict[str, Estimate]) -> None:
    fieldnames = [
        "run_id",
        "observable",
        "mean",
        "std",
        "naive_se",
        "block_se",
        "block_size_frames",
        "n_blocks",
        "tau_int_frames",
        "tau_se",
        "n_used_frames",
        "target_pressure_GPa",
        "target_density_g_cm3",
        "target_temperature_K",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, est in estimates.items():
            writer.writerow(
                {
                    "run_id": run_id,
                    "observable": name,
                    "mean": est.mean,
                    "std": est.std,
                    "naive_se": est.naive_se,
                    "block_se": est.block_se,
                    "block_size_frames": est.block_size,
                    "n_blocks": est.n_blocks,
                    "tau_int_frames": est.tau_int_frames,
                    "tau_se": est.tau_se,
                    "n_used_frames": n_used,
                    "target_pressure_GPa": settings.get("target_pressure_GPa", ""),
                    "target_density_g_cm3": settings.get("target_density_g_cm3", ""),
                    "target_temperature_K": settings.get("target_temperature_K", ""),
                }
            )


def plot_time_series(path: Path, rows: list[dict[str, float]], discard: int) -> None:
    time_ps = np.array([row["time_fs"] for row in rows]) / 1000.0
    series = [
        ("pressure_GPa", "Pressure (GPa)"),
        ("density_g_cm3", "Density (g/cm$^3$)"),
        ("temperature_K", "Temperature (K)"),
    ]

    fig, axes = plt.subplots(len(series), 1, figsize=(8.0, 7.0), sharex=True)
    for axis, (key, ylabel) in zip(axes, series):
        axis.plot(time_ps, [row[key] for row in rows], marker="o", linewidth=1.2, markersize=3)
        if discard > 0 and discard < len(rows):
            axis.axvline(time_ps[discard], color="0.25", linestyle="--", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (ps)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_eos_point(path: Path, rows: list[dict[str, float]], discard: int, estimates: dict[str, Estimate]) -> None:
    used = rows[discard:]
    density = np.array([row["density_g_cm3"] for row in used])
    pressure = np.array([row["pressure_GPa"] for row in used])
    temperature = np.array([row["temperature_K"] for row in used])

    fig, axis = plt.subplots(figsize=(6.0, 4.5))
    scatter = axis.scatter(density, pressure, c=temperature, cmap="viridis", s=36, alpha=0.85)
    axis.errorbar(
        estimates["density_g_cm3"].mean,
        estimates["pressure_GPa"].mean,
        xerr=estimates["density_g_cm3"].block_se,
        yerr=estimates["pressure_GPa"].block_se,
        fmt="s",
        color="black",
        ecolor="black",
        capsize=3,
        label="mean with block SE",
    )
    axis.set_xlabel("Density (g/cm$^3$)")
    axis.set_ylabel("Pressure (GPa)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Temperature (K)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(run_id: str, n_total: int, discard: int, settings: dict, estimates: dict[str, Estimate]) -> None:
    print(f"Run: {run_id}")
    print(f"Frames: {n_total} total, {n_total - discard} used after discarding {discard}")
    if settings:
        print("Targets from generated script:")
        for key in ("target_pressure_GPa", "target_density_g_cm3", "target_temperature_K"):
            if key in settings:
                print(f"  {key}: {settings[key]}")
    print("Observed means:")
    for key, est in estimates.items():
        se = est.block_se if math.isfinite(est.block_se) else est.tau_se
        se_label = "block SE" if math.isfinite(est.block_se) else "tau SE"
        print(
            f"  {key}: mean={est.mean:.8g}, std={est.std:.8g}, "
            f"naive_se={est.naive_se:.4g}, {se_label}={se:.4g}, "
            f"tau_int={est.tau_int_frames:.3g} saved frames"
        )
    if n_total - discard < 8:
        print(
            "Warning: fewer than 8 saved frames were used; uncertainty estimates "
            "are diagnostics only and are not production EOS error bars."
        )


def main() -> None:
    args = parse_args()
    xyz_path = args.xyz.resolve()
    output_dir = args.output_dir.resolve()
    if not xyz_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {xyz_path}")
    if args.discard < 0:
        raise ValueError("--discard must be non-negative.")

    rows = frame_table(xyz_path, args.timestep_fs, args.save_interval)
    if args.discard >= len(rows):
        raise ValueError(f"--discard={args.discard} leaves no frames from {len(rows)} saved frames.")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id_from_path(xyz_path)
    used = rows[args.discard :]
    settings = parse_run_settings(generated_script_path(xyz_path))
    estimates = {
        "pressure_GPa": estimate(row["pressure_GPa"] for row in used),
        "density_g_cm3": estimate(row["density_g_cm3"] for row in used),
        "temperature_K": estimate(row["temperature_K"] for row in used),
    }

    write_time_series_csv(output_dir / f"{run_id}_timeseries.csv", rows)
    write_summary_csv(output_dir / f"{run_id}_summary.csv", run_id, len(used), settings, estimates)
    with (output_dir / f"{run_id}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "trajectory": str(xyz_path),
                "discarded_frames": args.discard,
                "used_frames": len(used),
                "settings": settings,
                "estimates": {key: est.__dict__ for key, est in estimates.items()},
            },
            handle,
            indent=2,
        )

    plot_time_series(output_dir / f"{run_id}_timeseries.png", rows, args.discard)
    plot_eos_point(output_dir / f"{run_id}_eos_point.png", rows, args.discard, estimates)
    print_summary(run_id, len(rows), args.discard, settings, estimates)
    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
