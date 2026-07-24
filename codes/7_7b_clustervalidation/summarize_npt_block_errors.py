from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


DEFAULT_FOLDER = Path(r"C:\Users\shaoq\Documents\Mainz\mlip\outputsfull\r09_hot_w7n1")
AMU_TO_G = 1.66053906660e-24
ANG3_TO_CM3 = 1e-24
MASSES = {"H": 1.00784, "C": 12.011, "N": 14.0067, "O": 15.999, "S": 32.06}
OBSERVABLES = [
    ("temperature_K", "temperature", "K"),
    ("pressure_GPa", "pressure", "GPa"),
    ("density_g_cm3", "density", "g/cm^3"),
]


def find_one(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return files[0] if files else None


def load_thermo(path: Path) -> dict[str, np.ndarray]:
    header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].lstrip("#").split()
    data = np.atleast_2d(np.loadtxt(path))
    if data.shape[1] != len(header):
        header = [
            "time_fs",
            "temperature_K",
            "pressure_GPa",
            "energy_eV_per_atom",
            "kinetic_energy_eV_per_atom",
            "total_energy_eV_per_atom",
        ][: data.shape[1]]
    return dict(zip(header, data.T))


def xyz_densities(path: Path) -> np.ndarray:
    densities = []
    lattice_re = re.compile(r'Lattice="([^"]+)"')
    with path.open(encoding="utf-8", errors="ignore") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            comment = handle.readline()
            match = lattice_re.search(comment)
            symbols = [handle.readline().split()[0] for _ in range(natoms)]
            if not match:
                densities.append(np.nan)
                continue
            cell = np.array([float(x) for x in match.group(1).split()]).reshape(3, 3)
            mass_amu = sum(MASSES[symbol] for symbol in symbols)
            volume_ang3 = abs(np.linalg.det(cell))
            densities.append(mass_amu * AMU_TO_G / (volume_ang3 * ANG3_TO_CM3))
    return np.array(densities, dtype=float)


def load_run(folder: Path) -> tuple[Path, dict[str, np.ndarray]]:
    thermo_path = find_one(folder, "*thermo*.txt") or find_one(folder, "*.txt")
    if thermo_path is None:
        raise FileNotFoundError(f"No thermo text file found in {folder}")
    data = load_thermo(thermo_path)

    xyz_path = find_one(folder, f"{thermo_path.stem.replace('_thermo', '')}*.xyz") or find_one(folder, "*.xyz")
    if xyz_path is not None:
        densities = xyz_densities(xyz_path)
        n = min(len(next(iter(data.values()))), len(densities))
        data = {key: values[:n] for key, values in data.items()}
        data["density_g_cm3"] = densities[:n]
    return thermo_path, data


def production_data(data: dict[str, np.ndarray], cutoff_ps: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time_ps = data.get("time_fs", np.arange(len(next(iter(data.values()))))) / 1000.0
    mask = time_ps >= cutoff_ps
    if not np.any(mask):
        raise ValueError(f"Cutoff {cutoff_ps:g} ps leaves no production data.")
    return time_ps[mask], {key: values[mask] for key, values in data.items()}


def block_average(time_ps: np.ndarray, values: np.ndarray, block_ps: float) -> tuple[np.ndarray, int, float]:
    if len(time_ps) < 2:
        raise ValueError("Need at least two production frames for block averaging.")
    dt_ps = float(np.nanmedian(np.diff(time_ps)))
    frames_per_block = max(1, int(round(block_ps / dt_ps)))
    n_blocks = len(values) // frames_per_block
    if n_blocks < 2:
        raise ValueError(f"Only {n_blocks} complete {block_ps:g} ps blocks fit in production data.")
    used = n_blocks * frames_per_block
    blocks = values[:used].reshape(n_blocks, frames_per_block)
    return np.nanmean(blocks, axis=1), frames_per_block, dt_ps * frames_per_block


def summarize_blocks(block_values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.nanmean(block_values))
    std = float(np.nanstd(block_values, ddof=1))
    stderr = std / np.sqrt(len(block_values))
    return mean, std, stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate NPT thermo uncertainties from uncorrelated block means.")
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--cutoff-ps", type=float, default=25.0, help="Initial transient cutoff in ps.")
    parser.add_argument("--block-ps", type=float, default=3.0, help="Block length in ps.")
    args = parser.parse_args()

    thermo_path, data = load_run(args.folder)
    time_ps, prod = production_data(data, args.cutoff_ps)
    rows = []

    for key, label, unit in OBSERVABLES:
        if key not in prod:
            raise KeyError(f"Missing required observable {key!r}.")
        block_values, frames_per_block, actual_block_ps = block_average(time_ps, np.asarray(prod[key], dtype=float), args.block_ps)
        mean, std, stderr = summarize_blocks(block_values)
        rows.append({
            "observable": label,
            "unit": unit,
            "mean": mean,
            "block_std": std,
            "standard_error": stderr,
            "n_blocks": len(block_values),
            "frames_per_block": frames_per_block,
            "actual_block_ps": actual_block_ps,
        })

    out_path = thermo_path.parent / "block_error_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input thermo: {thermo_path}")
    print(f"Production starts: {args.cutoff_ps:g} ps")
    print(f"Requested block length: {args.block_ps:g} ps")
    for row in rows:
        print(
            f"{row['observable']:>11s}: {row['mean']:.6g} +/- {row['standard_error']:.3g} {row['unit']} "
            f"(block std {row['block_std']:.3g}, n={row['n_blocks']}, block={row['actual_block_ps']:.3g} ps)"
        )
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
