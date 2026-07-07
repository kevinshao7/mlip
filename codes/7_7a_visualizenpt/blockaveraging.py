from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FOLDER = Path(r"C:\Users\shaoq\Documents\Mainz\mlip\outputsfull\r09_hot_w")
OUT_DIR = Path(__file__).resolve().parent
AMU_TO_G = 1.66053906660e-24
ANG3_TO_CM3 = 1e-24
MASSES = {"H": 1.00784, "C": 12.011, "N": 14.0067, "O": 15.999, "S": 32.06}
PLOT_KEYS = [
    ("temperature_K", "Temperature (K)"),
    ("pressure_GPa", "Pressure (GPa)"),
    ("density_g_cm3", "Density (g/cm^3)"),
    ("energy_eV_per_atom", "Potential energy (eV/atom)"),
    ("kinetic_energy_eV_per_atom", "Kinetic energy (eV/atom)"),
    ("total_energy_eV_per_atom", "Total energy (eV/atom)"),
]


def find_one(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return files[0] if files else None


def load_thermo(path: Path) -> dict[str, np.ndarray]:
    header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].lstrip("#").split()
    data = np.atleast_2d(np.loadtxt(path))
    if data.shape[1] != len(header):
        header = ["time_fs", "temperature_K", "pressure_GPa", "energy_eV_per_atom",
                  "kinetic_energy_eV_per_atom", "total_energy_eV_per_atom"][: data.shape[1]]
    return dict(zip(header, data.T))


def xyz_densities(path: Path) -> np.ndarray:
    densities = []
    lattice_re = re.compile(r'Lattice="([^"]+)"')
    with path.open(encoding="utf-8", errors="ignore") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            match = lattice_re.search(fh.readline())
            symbols = [fh.readline().split()[0] for _ in range(natoms)]
            if not match:
                densities.append(np.nan)
                continue
            cell = np.array([float(x) for x in match.group(1).split()]).reshape(3, 3)
            mass = sum(MASSES.get(sym, np.nan) for sym in symbols)
            densities.append(mass * AMU_TO_G / (abs(np.linalg.det(cell)) * ANG3_TO_CM3))
    return np.array(densities, dtype=float)


def load_run(folder: Path) -> tuple[Path, dict[str, np.ndarray]]:
    txt = find_one(folder, "*thermo*.txt") or find_one(folder, "*.txt")
    if txt is None:
        raise FileNotFoundError(f"No .txt file found in {folder}")
    data = load_thermo(txt)

    xyz = find_one(folder, f"{txt.stem.replace('_thermo', '')}*.xyz") or find_one(folder, "*.xyz")
    if xyz:
        density = xyz_densities(xyz)
        n = min(len(next(iter(data.values()))), len(density))
        data = {key: val[:n] for key, val in data.items()}
        data["density_g_cm3"] = density[:n]
    return txt, data


def autocorr_fft(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    n = len(x)
    if n < 2 or np.allclose(x, 0.0):
        return np.ones(max(n, 1))
    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, fft_len)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_len)[:n]
    acf /= np.arange(n, 0, -1)
    return np.real(acf / acf[0])


def block_means(time_ps: np.ndarray, values: np.ndarray, target_blocks: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    n = len(values)
    n_blocks = min(target_blocks, n)
    block_size = max(1, n // n_blocks)
    used = n_blocks * block_size
    centers = time_ps[:used].reshape(n_blocks, block_size).mean(axis=1)
    means = values[:used].reshape(n_blocks, block_size).mean(axis=1)
    block_ps = float(np.nanmean(np.diff(time_ps)) * block_size) if len(time_ps) > 1 else 0.0
    return centers, means, block_size, block_ps


def production_slice(data: dict[str, np.ndarray], cutoff_ps: float) -> tuple[np.ndarray, np.ndarray]:
    time_ps = data.get("time_fs", np.arange(len(next(iter(data.values()))))) / 1000.0
    mask = time_ps >= cutoff_ps
    if not np.any(mask):
        raise ValueError(f"Cutoff {cutoff_ps:g} ps leaves no production data.")
    return time_ps, mask


def plot_timeseries(txt: Path, data: dict[str, np.ndarray], cutoff_ps: float) -> Path:
    time_ps, mask = production_slice(data, cutoff_ps)
    keys = [(key, label) for key, label in PLOT_KEYS if key in data]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.2 * len(keys)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (key, label) in zip(axes, keys):
        y = np.asarray(data[key], dtype=float)
        prod = y[mask]
        mean = np.nanmean(prod)
        std = np.nanstd(prod, ddof=1)
        ax.plot(time_ps, y, linewidth=1.0)
        ax.axvline(cutoff_ps, color="black", linestyle="--", linewidth=1.0, label=f"cutoff {cutoff_ps:g} ps")
        ax.axhspan(mean - std, mean + std, color="0.8", alpha=0.6, label="production mean +/- 1 std")
        ax.axhline(mean, color="black", linewidth=1.0)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time (ps)")
    out = OUT_DIR / f"{txt.stem}_cutoff_timeseries.png"
    fig.suptitle(f"{txt.parent.name}: production starts at {cutoff_ps:g} ps")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_blocks(txt: Path, data: dict[str, np.ndarray], cutoff_ps: float) -> Path:
    time_ps, mask = production_slice(data, cutoff_ps)
    prod_time = time_ps[mask]
    keys = [(key, label) for key, label in PLOT_KEYS if key in data]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.4 * len(keys)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (key, label) in zip(axes, keys):
        prod = np.asarray(data[key], dtype=float)[mask]
        mean = np.nanmean(prod)
        std = np.nanstd(prod, ddof=1)
        ax.axhspan(mean - std, mean + std, color="0.8", alpha=0.6, label="naive +/- 1 std")
        ax.axhline(mean, color="black", linewidth=1.0, label="production mean")
        for target in (10, 30, 100):
            centers, means, block_size, block_ps = block_means(prod_time, prod, target)
            label_text = f"{target} blocks: {block_size} frames, {block_ps:.2g} ps/block"
            ax.plot(centers, means, marker="o", linewidth=1.0, markersize=2.5, label=label_text)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("Production time (ps)")
    out = OUT_DIR / f"{txt.stem}_block_averages.png"
    fig.suptitle(f"Block averages after {cutoff_ps:g} ps cutoff")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_acf(txt: Path, data: dict[str, np.ndarray], cutoff_ps: float) -> Path:
    time_ps, mask = production_slice(data, cutoff_ps)
    dt_ps = float(np.nanmean(np.diff(time_ps[mask]))) if np.count_nonzero(mask) > 1 else 1.0
    keys = [(key, label) for key, label in PLOT_KEYS if key in data]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.2 * len(keys)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (key, label) in zip(axes, keys):
        acf = autocorr_fft(np.asarray(data[key], dtype=float)[mask])
        lag_ps = np.arange(len(acf)) * dt_ps
        ax.plot(lag_ps, acf, linewidth=1.1)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Lag time (ps)")
    out = OUT_DIR / f"{txt.stem}_autocorrelation.png"
    fig.suptitle(f"FFT autocorrelation after {cutoff_ps:g} ps cutoff")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-average NPT thermo output after an initial cutoff.")
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--cutoff-ps", type=float, default=25.0, help="Initial transient cutoff in ps.")
    args = parser.parse_args()

    txt, data = load_run(args.folder)
    outputs = [
        plot_timeseries(txt, data, args.cutoff_ps),
        plot_blocks(txt, data, args.cutoff_ps),
        plot_acf(txt, data, args.cutoff_ps),
    ]
    for out in outputs:
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
