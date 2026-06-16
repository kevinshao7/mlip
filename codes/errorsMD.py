#!/usr/bin/env python3
"""Estimate MD autocorrelations and correlated statistical errors.

This script implements the Allen/Tildesley Chapter 8 workflow:

* autocorrelation functions via FFT with zero padding,
* integrated correlation times and statistical inefficiency,
* block and bootstrap estimates of uncertainties, including uncertainty in
  the autocorrelation-derived error estimate itself.

It intentionally avoids ASE so it can read the existing extended XYZ files in
this repository from their headers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-errorsMD")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THERMO_COLUMNS = ("time_fs", "temperature_K", "pressure_GPa", "energy_eV_per_atom")
XYZ_OBSERVABLES = (
    "xyz_energy_eV_per_atom",
    "xyz_pressure_GPa",
    "xyz_dipole_norm_per_atom",
)
PRESSURE_EV_A3_TO_GPA = 160.21766208

OBSERVABLE_LABELS = {
    "temperature_K": "temperature (K)",
    "pressure_GPa": "pressure (GPa)",
    "energy_eV_per_atom": "potential energy per atom (eV/atom)",
    "xyz_energy_eV_per_atom": "potential energy per atom (eV/atom)",
    "xyz_pressure_GPa": "pressure (GPa)",
    "xyz_dipole_norm_per_atom": "dipole norm per atom",
}


@dataclass
class SeriesAnalysis:
    name: str
    n: int
    dt_fs: float
    mean: float
    std: float
    naive_stderr: float
    tau_int_fs: float
    tau_int_stderr_fs: float
    statistical_inefficiency: float
    statistical_inefficiency_stderr: float
    real_stderr: float
    real_stderr_stderr: float
    block_stderr: float
    block_size: int
    n_blocks: int
    cutoff_lag: int
    tail_start_lag: int | None
    tail_tau_fs: float | None
    tail_amplitude: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FFT autocorrelation and correlated error analysis for MD outputs."
    )
    parser.add_argument(
        "--thermo",
        nargs="*",
        default=[
            "mace_1500K_density_1.5_thermo.txt",
            "mace_1500K_density_2.0_thermo.txt",
        ],
        help="Thermo text files with time, temperature, pressure, energy columns.",
    )
    parser.add_argument(
        "--xyz",
        nargs="*",
        default=[
            "mace_1500K_density_1.5.xyz",
            "mace_1500K_density_2.0.xyz",
            "mace01_md.xyz",
        ],
        help="Extended XYZ files to sample per-frame header observables.",
    )
    parser.add_argument(
        "--drop-fraction",
        type=float,
        default=0.2,
        help="Initial fraction of each trajectory to discard as equilibration.",
    )
    parser.add_argument(
        "--max-lag-fraction",
        type=float,
        default=0.5,
        help="Maximum autocorrelation lag as a fraction of retained samples.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Moving-block bootstrap replicates for ACF/error uncertainty.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/errorsMD_summary.json",
        help="Path for machine-readable summary.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run numerical sanity checks and exit.",
    )
    return parser.parse_args()


def safe_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "plot"


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def infer_dt_fs(time_fs: np.ndarray) -> float:
    if len(time_fs) < 2:
        return 1.0
    diffs = np.diff(time_fs)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def fft_autocovariance(values: np.ndarray) -> np.ndarray:
    """Unbiased autocovariance C(k) = sum_t dA_t dA_{t+k} / (N-k)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("Need at least two finite samples for autocorrelation")
    fluctuations = values - np.mean(values)
    n = len(fluctuations)
    nfft = next_power_of_two(2 * n)
    transformed = np.fft.rfft(fluctuations, n=nfft)
    raw = np.fft.irfft(transformed * np.conjugate(transformed), n=nfft)[:n]
    norm = np.arange(n, 0, -1, dtype=float)
    return raw / norm


def direct_autocovariance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fluctuations = values - np.mean(values)
    n = len(fluctuations)
    return np.asarray(
        [
            np.dot(fluctuations[: n - lag], fluctuations[lag:]) / (n - lag)
            for lag in range(n)
        ],
        dtype=float,
    )


def normalized_acf(values: np.ndarray) -> np.ndarray:
    cov = fft_autocovariance(values)
    if not np.isfinite(cov[0]) or cov[0] <= 0:
        raise ValueError("Cannot normalize an autocorrelation with zero variance")
    return cov / cov[0]


def find_cutoff_lag(acf: np.ndarray, c: float = 5.0) -> int:
    """Self-consistent window inspired by Sokal's automatic window rule."""
    running_tau_steps = 0.5 + np.cumsum(acf[1:])
    for lag in range(1, len(acf)):
        tau_steps = max(float(running_tau_steps[lag - 1]), 0.5)
        if lag >= c * tau_steps:
            return lag
    return len(acf) - 1


def fit_exponential_tail(
    acf: np.ndarray, dt_fs: float, cutoff_lag: int
) -> tuple[int | None, float | None, float | None, float]:
    """Fit positive long-time ACF values to A exp(-t/tau).

    Returns tail_start_lag, amplitude, tau_fs, integral_tail_steps.
    """
    if cutoff_lag < 8:
        return None, None, None, 0.0
    lo = max(3, cutoff_lag // 3)
    hi = cutoff_lag
    x = np.arange(lo, hi + 1, dtype=float)
    y = acf[lo : hi + 1]
    mask = np.isfinite(y) & (y > 0.0) & (y < 0.95)
    if np.count_nonzero(mask) < 4:
        return None, None, None, 0.0

    xfit = x[mask]
    yfit = y[mask]
    slope, intercept = np.polyfit(xfit * dt_fs, np.log(yfit), deg=1)
    if slope >= 0:
        return None, None, None, 0.0
    tau_fs = -1.0 / float(slope)
    amplitude = float(math.exp(intercept))
    tail_start = int(xfit[-1]) + 1
    if tau_fs <= 0 or not np.isfinite(tau_fs):
        return None, None, None, 0.0

    tail_t_fs = tail_start * dt_fs
    # Integral in units of stored-sample steps, for use in s = 1 + 2 sum c(k).
    tail_steps = amplitude * math.exp(-tail_t_fs / tau_fs) * (tau_fs / dt_fs)
    return tail_start, amplitude, tau_fs, max(float(tail_steps), 0.0)


def integrated_correlation(
    acf: np.ndarray, dt_fs: float
) -> tuple[float, float, int, int | None, float | None, float | None]:
    cutoff_lag = find_cutoff_lag(acf)
    tail_start, amplitude, tail_tau_fs, tail_steps = fit_exponential_tail(
        acf, dt_fs, cutoff_lag
    )
    observed_sum = float(np.sum(acf[1 : cutoff_lag + 1]))
    tau_steps = max(0.5 + observed_sum + tail_steps, 0.5)
    tau_fs = tau_steps * dt_fs
    statistical_inefficiency = max(2.0 * tau_steps, 1.0)
    return (
        tau_fs,
        statistical_inefficiency,
        cutoff_lag,
        tail_start,
        tail_tau_fs,
        amplitude,
    )


def blocking_stderr(values: np.ndarray) -> tuple[float, int, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    current = values.copy()
    best_stderr = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    best_block_size = 1
    best_n_blocks = len(values)

    block_size = 1
    previous = None
    while len(current) >= 8:
        n_blocks = len(current)
        stderr = float(np.std(current, ddof=1) / math.sqrt(n_blocks))
        if previous is None or stderr >= 0.8 * previous:
            best_stderr = stderr
            best_block_size = block_size
            best_n_blocks = n_blocks
        previous = stderr
        if len(current) % 2 == 1:
            current = current[:-1]
        current = 0.5 * (current[0::2] + current[1::2])
        block_size *= 2

    return best_stderr, best_block_size, best_n_blocks


def bootstrap_integrated_errors(
    values: np.ndarray,
    dt_fs: float,
    block_size: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Moving-block bootstrap for tau_int, s, and real stderr."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n_bootstrap <= 1 or n < 16:
        return float("nan"), float("nan"), float("nan")
    block_size = int(np.clip(block_size, 2, max(2, n // 2)))
    starts = np.arange(0, n - block_size + 1)
    if len(starts) == 0:
        return float("nan"), float("nan"), float("nan")

    tau_values = []
    s_values = []
    stderr_values = []
    for _ in range(n_bootstrap):
        pieces = []
        while sum(len(piece) for piece in pieces) < n:
            start = int(rng.choice(starts))
            pieces.append(values[start : start + block_size])
        sample = np.concatenate(pieces)[:n]
        try:
            acf = normalized_acf(sample)
            tau_fs, s, *_ = integrated_correlation(acf, dt_fs)
        except ValueError:
            continue
        variance = float(np.var(sample, ddof=1))
        tau_values.append(tau_fs)
        s_values.append(s)
        stderr_values.append(math.sqrt(max(s, 1.0) * variance / len(sample)))

    if len(tau_values) < 4:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.std(tau_values, ddof=1)),
        float(np.std(s_values, ddof=1)),
        float(np.std(stderr_values, ddof=1)),
    )


def parse_extended_xyz_headers(path: Path) -> dict[str, np.ndarray]:
    rows = {name: [] for name in XYZ_OBSERVABLES}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue
            natoms = int(natoms_line)
            header = handle.readline().strip()
            fields = {}
            for token in shlex.split(header):
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value

            energy = float(fields.get("energy", "nan"))
            rows["xyz_energy_eV_per_atom"].append(energy / natoms)

            stress = parse_float_list(fields.get("stress", ""))
            if len(stress) >= 9:
                pressure = -float(np.mean([stress[0], stress[4], stress[8]]))
                rows["xyz_pressure_GPa"].append(pressure * PRESSURE_EV_A3_TO_GPA)
            else:
                rows["xyz_pressure_GPa"].append(float("nan"))

            dipole = parse_float_list(fields.get("dipole", ""))
            if len(dipole) >= 3:
                rows["xyz_dipole_norm_per_atom"].append(
                    float(np.linalg.norm(dipole[:3])) / natoms
                )
            else:
                rows["xyz_dipole_norm_per_atom"].append(float("nan"))

            for _ in range(natoms):
                handle.readline()

    return {name: np.asarray(values, dtype=float) for name, values in rows.items()}


def parse_float_list(text: str) -> list[float]:
    values = []
    for part in text.replace(",", " ").split():
        try:
            values.append(float(part))
        except ValueError:
            pass
    return values


def load_thermo(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    time = data[:, 0]
    series = {
        "temperature_K": data[:, 1],
        "pressure_GPa": data[:, 2],
        "energy_eV_per_atom": data[:, 3],
    }
    return time, series


def retained_time_and_values(
    time_fs: np.ndarray | None,
    values: np.ndarray,
    dt_fs: float,
    drop_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if time_fs is None or len(time_fs) != len(values):
        time_fs = np.arange(len(values), dtype=float) * dt_fs
    else:
        time_fs = np.asarray(time_fs, dtype=float)
    finite = np.isfinite(values) & np.isfinite(time_fs)
    values = values[finite]
    time_fs = time_fs[finite]
    drop = int(len(values) * drop_fraction)
    return time_fs[drop:], values[drop:]


def y_label_for(name: str) -> str:
    for suffix, label in sorted(OBSERVABLE_LABELS.items(), key=lambda item: -len(item[0])):
        if name.endswith(suffix):
            return label
    return "value"


def analyze_series(
    name: str,
    time_fs: np.ndarray | None,
    values: np.ndarray,
    dt_fs: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
    plots_dir: Path,
) -> SeriesAnalysis | None:
    time_fs, values = retained_time_and_values(time_fs, values, dt_fs, args.drop_fraction)
    if len(values) < 16:
        print(f"Skipping {name}: only {len(values)} finite retained samples")
        return None

    max_lag = max(4, int(len(values) * args.max_lag_fraction))
    acf = normalized_acf(values)[:max_lag]
    tau_fs, s, cutoff_lag, tail_start, tail_tau_fs, tail_amplitude = (
        integrated_correlation(acf, dt_fs)
    )
    variance = float(np.var(values, ddof=1))
    std = math.sqrt(variance)
    naive_stderr = std / math.sqrt(len(values))
    real_stderr = math.sqrt(max(s, 1.0) * variance / len(values))

    block_stderr, block_size, n_blocks = blocking_stderr(values)
    tau_se, s_se, real_stderr_se = bootstrap_integrated_errors(
        values, dt_fs, block_size, args.bootstrap, rng
    )

    block_acf_mean, block_acf_se = block_acf_errors(values, max_lag, block_size)
    plot_title = safe_label(name)
    make_plots(
        plots_dir=plots_dir,
        plot_title=plot_title,
        display_name=name,
        time_fs=time_fs,
        values=values,
        y_label=y_label_for(name),
        dt_fs=dt_fs,
        acf=acf,
        block_acf_mean=block_acf_mean,
        block_acf_se=block_acf_se,
        cutoff_lag=cutoff_lag,
        tail_start=tail_start,
        tail_tau_fs=tail_tau_fs,
        tail_amplitude=tail_amplitude,
        s=s,
        block_stderr=block_stderr,
        real_stderr=real_stderr,
    )

    return SeriesAnalysis(
        name=name,
        n=len(values),
        dt_fs=dt_fs,
        mean=float(np.mean(values)),
        std=std,
        naive_stderr=naive_stderr,
        tau_int_fs=tau_fs,
        tau_int_stderr_fs=tau_se,
        statistical_inefficiency=s,
        statistical_inefficiency_stderr=s_se,
        real_stderr=real_stderr,
        real_stderr_stderr=real_stderr_se,
        block_stderr=block_stderr,
        block_size=block_size,
        n_blocks=n_blocks,
        cutoff_lag=cutoff_lag,
        tail_start_lag=tail_start,
        tail_tau_fs=tail_tau_fs,
        tail_amplitude=tail_amplitude,
    )


def block_acf_errors(
    values: np.ndarray, max_lag: int, block_size_hint: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    n = len(values)
    min_block = max(16, 4 * max(block_size_hint, 1), max_lag // 2)
    n_blocks = n // min_block
    if n_blocks < 3:
        return None, None
    acfs = []
    for i in range(n_blocks):
        block = values[i * min_block : (i + 1) * min_block]
        try:
            acfs.append(normalized_acf(block)[:max_lag])
        except ValueError:
            continue
    if len(acfs) < 3:
        return None, None
    min_len = min(len(acf) for acf in acfs)
    stack = np.vstack([acf[:min_len] for acf in acfs])
    return np.mean(stack, axis=0), np.std(stack, axis=0, ddof=1) / math.sqrt(len(acfs))


def make_plots(
    plots_dir: Path,
    plot_title: str,
    display_name: str,
    time_fs: np.ndarray,
    values: np.ndarray,
    y_label: str,
    dt_fs: float,
    acf: np.ndarray,
    block_acf_mean: np.ndarray | None,
    block_acf_se: np.ndarray | None,
    cutoff_lag: int,
    tail_start: int | None,
    tail_tau_fs: float | None,
    tail_amplitude: float | None,
    s: float,
    block_stderr: float,
    real_stderr: float,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    lags = np.arange(len(acf))
    t_fs = lags * dt_fs

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_fs, values, lw=1.2)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Time series: {display_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"errorsMDplot_{plot_title}_timeseries.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_fs, acf, label="FFT autocorrelation", lw=1.6)
    if block_acf_mean is not None and block_acf_se is not None:
        bt = np.arange(len(block_acf_mean)) * dt_fs
        ax.fill_between(
            bt,
            block_acf_mean - 2.0 * block_acf_se,
            block_acf_mean + 2.0 * block_acf_se,
            alpha=0.2,
            label="block +/- 2 SE",
        )
    if cutoff_lag < len(acf):
        ax.axvline(cutoff_lag * dt_fs, color="k", ls="--", lw=1, label="integration cutoff")
    if tail_start is not None and tail_tau_fs is not None and tail_amplitude is not None:
        tail_t = t_fs[tail_start:]
        if len(tail_t) > 0:
            fit = tail_amplitude * np.exp(-tail_t / tail_tau_fs)
            ax.plot(tail_t, fit, color="tab:red", lw=1.2, label="tail fit")
    ax.set_xlabel("lag (fs)")
    ax.set_ylabel("c(t)")
    ax.set_title(f"Autocorrelation: {display_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"errorsMDplot_{plot_title}_autocorrelation.png", dpi=200)
    plt.close(fig)

    running_s = 1.0 + 2.0 * np.cumsum(acf[1:])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_fs[1:], running_s, lw=1.6)
    ax.axhline(s, color="tab:red", ls="--", lw=1, label=f"chosen s={s:.2g}")
    if cutoff_lag < len(acf):
        ax.axvline(cutoff_lag * dt_fs, color="k", ls="--", lw=1)
    ax.set_xlabel("integration limit (fs)")
    ax.set_ylabel("statistical inefficiency s")
    ax.set_title(f"Running inefficiency: {display_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"errorsMDplot_{plot_title}_statistical_inefficiency.png", dpi=200)
    plt.close(fig)

    block_sizes, block_errors = blocking_curve(values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(block_sizes, block_errors, marker="o", lw=1.4, label="blocking")
    ax.axhline(real_stderr, color="tab:red", ls="--", lw=1, label="ACF error")
    ax.axhline(block_stderr, color="tab:green", ls=":", lw=1, label="selected block error")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block size (stored frames)")
    ax.set_ylabel(f"standard error of mean {y_label}")
    ax.set_title(f"Correlated mean error: {display_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"errorsMDplot_{plot_title}_blocking_error.png", dpi=200)
    plt.close(fig)


def blocking_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(values, dtype=float).copy()
    block_size = 1
    block_sizes = []
    errors = []
    while len(current) >= 4:
        block_sizes.append(block_size)
        errors.append(float(np.std(current, ddof=1) / math.sqrt(len(current))))
        if len(current) % 2 == 1:
            current = current[:-1]
        current = 0.5 * (current[0::2] + current[1::2])
        block_size *= 2
    return np.asarray(block_sizes), np.asarray(errors)


def as_jsonable(analysis: SeriesAnalysis) -> dict[str, float | int | str | None]:
    output = {}
    for field in SeriesAnalysis.__dataclass_fields__:
        value = getattr(analysis, field)
        if isinstance(value, float) and not math.isfinite(value):
            output[field] = None
        else:
            output[field] = value
    return output


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    test = rng.normal(size=257)
    fft_cov = fft_autocovariance(test)
    direct_cov = direct_autocovariance(test)
    max_error = float(np.max(np.abs(fft_cov - direct_cov)))
    if max_error > 1.0e-10:
        raise AssertionError(f"FFT autocovariance differs from direct: {max_error}")

    ar = np.empty(512)
    ar[0] = rng.normal()
    for i in range(1, len(ar)):
        ar[i] = 0.85 * ar[i - 1] + rng.normal(scale=0.5)
    acf = normalized_acf(ar)
    tau_fs, s, cutoff_lag, *_ = integrated_correlation(acf, dt_fs=5.0)
    if tau_fs <= 0.0 or s < 1.0 or cutoff_lag <= 0:
        raise AssertionError("Integrated correlation estimates are not physical")

    try:
        normalized_acf(np.ones(16))
    except ValueError:
        pass
    else:
        raise AssertionError("Constant series should not have a normalized ACF")

    print("Self-test passed")
    print(f"FFT/direct max abs error: {max_error:.3e}")
    print(f"AR(1) sanity: tau={tau_fs:.3g} fs, s={s:.3g}, cutoff={cutoff_lag}")


def analyze_thermo_files(
    paths: Iterable[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
    plots_dir: Path,
) -> list[SeriesAnalysis]:
    analyses = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            print(f"Missing thermo file, skipping: {path}")
            continue
        time_fs, series = load_thermo(path)
        dt_fs = infer_dt_fs(time_fs)
        for column, values in series.items():
            name = f"{path.stem}_{column}"
            analysis = analyze_series(name, time_fs, values, dt_fs, args, rng, plots_dir)
            if analysis is not None:
                analyses.append(analysis)
    return analyses


def analyze_xyz_files(
    paths: Iterable[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
    plots_dir: Path,
) -> list[SeriesAnalysis]:
    analyses = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            print(f"Missing xyz file, skipping: {path}")
            continue
        try:
            series = parse_extended_xyz_headers(path)
        except (OSError, ValueError) as exc:
            print(f"Could not parse {path}: {exc}")
            continue
        thermo_guess = path.with_name(path.stem + "_thermo.txt")
        if thermo_guess.exists():
            time_fs, _ = load_thermo(thermo_guess)
            dt_fs = infer_dt_fs(time_fs)
        else:
            time_fs = None
            dt_fs = 1.0
        for observable, values in series.items():
            if np.count_nonzero(np.isfinite(values)) < 16:
                continue
            name = f"{path.stem}_{observable}"
            analysis = analyze_series(name, time_fs, values, dt_fs, args, rng, plots_dir)
            if analysis is not None:
                analyses.append(analysis)
    return analyses


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not 0.0 <= args.drop_fraction < 0.9:
        raise ValueError("--drop-fraction must be in [0, 0.9)")
    if not 0.01 <= args.max_lag_fraction <= 1.0:
        raise ValueError("--max-lag-fraction must be in [0.01, 1.0]")

    plots_dir = Path(args.plots_dir)
    rng = np.random.default_rng(args.seed)

    analyses = []
    analyses.extend(analyze_thermo_files(args.thermo, args, rng, plots_dir))
    analyses.extend(analyze_xyz_files(args.xyz, args, rng, plots_dir))

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump([as_jsonable(analysis) for analysis in analyses], handle, indent=2)

    print(f"Analyzed {len(analyses)} series")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plots to: {plots_dir}")
    for analysis in analyses:
        print(
            f"{analysis.name}: mean={analysis.mean:.6g}, "
            f"stderr={analysis.real_stderr:.3g}, "
            f"s={analysis.statistical_inefficiency:.3g}, "
            f"tau={analysis.tau_int_fs:.3g} fs"
        )


if __name__ == "__main__":
    main()
