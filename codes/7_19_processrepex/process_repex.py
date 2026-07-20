from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(r"C:\Users\shaoq\Documents\Mainz\mlip\outputsfull\7_20_repex")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "plots"

QUANTITIES = (
    ("density", "Density", r"g cm$^{-3}$"),
    ("temperature", "Temperature", "K"),
    ("total_energy", "Total energy", "kJ/mol"),
)

FS_PER_PS = 1000.0


def replica_number(path: Path) -> int:
    """Return the integer in replica_<integer> for natural sorting."""
    match = re.match(r"replica_(\d+)(?:_|$)", path.name)
    return int(match.group(1)) if match else 10**9


def load_thermo(path: Path) -> dict[str, np.ndarray]:
    """Read a thermo.csv whose comma-separated header may start with '#'."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"Empty thermo file: {path}")

    columns = next(csv.reader([first_line.lstrip("# ")]))
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] != len(columns):
        raise ValueError(
            f"{path}: header has {len(columns)} columns but data has {data.shape[1]}"
        )
    return {name.strip(): data[:, i] for i, name in enumerate(columns)}


def autocorrelation(values: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Calculate the normalized, unbiased autocorrelation using an FFT."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.ones(values.size, dtype=float)

    centered = values - values.mean()
    variance_sum = np.dot(centered, centered)
    if variance_sum <= np.finfo(float).eps:
        # A constant signal has no meaningful correlation beyond lag zero.
        result = np.zeros(values.size, dtype=float)
        result[0] = 1.0
    else:
        fft_size = 1 << (2 * values.size - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=fft_size)
        covariance = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)[: values.size]
        covariance /= np.arange(values.size, 0, -1)
        result = covariance / covariance[0]

    if max_lag is not None:
        result = result[: min(max_lag + 1, result.size)]
    return result


def plot_replica(thermo_path: Path, output_dir: Path, max_lag: int | None) -> Path:
    thermo = load_thermo(thermo_path)
    missing = [column for column, _, _ in QUANTITIES if column not in thermo]
    if missing:
        raise KeyError(f"{thermo_path}: missing columns: {', '.join(missing)}")

    sample_count = len(thermo[QUANTITIES[0][0]])
    plot_max_lag = sample_count // 2 if max_lag is None else max_lag
    time = thermo.get("time", np.arange(sample_count, dtype=float)) / FS_PER_PS
    if time.size > 1:
        sample_interval = float(np.nanmedian(np.diff(time)))
        if not np.isfinite(sample_interval) or sample_interval <= 0:
            sample_interval = 1.0
    else:
        sample_interval = 1.0

    fig, axes = plt.subplots(
        len(QUANTITIES), 2, figsize=(13, 8), constrained_layout=True
    )
    for row, (column, title, unit) in enumerate(QUANTITIES):
        values = thermo[column]
        finite = np.isfinite(time) & np.isfinite(values)
        axes[row, 0].plot(time[finite], values[finite], linewidth=0.8)
        axes[row, 0].set_ylabel(f"{title} ({unit})")

        acf = autocorrelation(values, max_lag=plot_max_lag)
        lag_time = np.arange(acf.size) * sample_interval
        axes[row, 1].plot(lag_time, acf, linewidth=0.9)
        axes[row, 1].axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        axes[row, 1].set_ylabel("Autocorrelation")
        axes[row, 1].set_ylim(-1.05, 1.05)

        for ax in axes[row]:
            ax.grid(alpha=0.25)

    axes[0, 0].set_title("Time series")
    axes[0, 1].set_title("Normalized autocorrelation")
    axes[-1, 0].set_xlabel("Time (ps)")
    axes[-1, 1].set_xlabel("Lag time (ps)")
    fig.suptitle(thermo_path.parent.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{thermo_path.parent.name}_thermo_diagnostics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def process_all(input_dir: Path, output_dir: Path, max_lag: int | None) -> list[Path]:
    replicas = sorted(
        (path for path in input_dir.glob("replica_*") if path.is_dir()),
        key=replica_number,
    )
    if not replicas:
        raise FileNotFoundError(f"No replica_* directories found in {input_dir}")

    outputs: list[Path] = []
    for replica in replicas:
        thermo_path = replica / "thermo.csv"
        if not thermo_path.is_file():
            print(f"Skipping {replica.name}: thermo.csv not found")
            continue
        output = plot_replica(thermo_path, output_dir, max_lag)
        outputs.append(output)
        print(f"Saved {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot thermo time series and autocorrelations for replica-exchange runs."
    )
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Maximum autocorrelation lag in samples (default: half the trajectory).",
    )
    args = parser.parse_args()
    if args.max_lag is not None and args.max_lag < 0:
        parser.error("--max-lag must be non-negative")

    outputs = process_all(args.input_dir, args.output_dir, args.max_lag)
    print(f"Created {len(outputs)} figure(s).")


if __name__ == "__main__":
    main()
