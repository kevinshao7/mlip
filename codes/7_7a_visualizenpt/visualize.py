from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FOLDER = Path(r"C:\Users\shaoq\Documents\Mainz\mlip\outputsfull\r09_hot_w")
AMU_TO_G = 1.66053906660e-24
ANG3_TO_CM3 = 1e-24
MASSES = {"H": 1.00784, "C": 12.011, "N": 14.0067, "O": 15.999, "S": 32.06}


def find_one(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return files[0] if files else None


def load_thermo(path: Path) -> dict[str, np.ndarray]:
    header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].lstrip("#").split()
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
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
            volume = abs(np.linalg.det(cell))
            mass = sum(MASSES.get(sym, np.nan) for sym in symbols)
            densities.append(mass * AMU_TO_G / (volume * ANG3_TO_CM3))
    return np.array(densities, dtype=float)


def plot(folder: Path) -> Path:
    txt = find_one(folder, "*thermo*.txt") or find_one(folder, "*.txt")
    if txt is None:
        raise FileNotFoundError(f"No .txt file found in {folder}")

    thermo = load_thermo(txt)
    time = thermo.get("time_fs", np.arange(len(next(iter(thermo.values())))))
    xyz = find_one(folder, f"{txt.stem.replace('_thermo', '')}*.xyz") or find_one(folder, "*.xyz")
    if xyz:
        density = xyz_densities(xyz)
        n = min(len(time), len(density))
        thermo["density_g_cm3"] = density[:n]
        time = time[:n]
        thermo = {key: val[:n] for key, val in thermo.items()}
    panels = [
        ("temperature_K", "Temperature (K)", False),
        ("pressure_GPa", "Pressure (GPa)", False),
        ("density_g_cm3", "Density (g/cm^3)", False),
    ]
    panels = [panel for panel in panels if panel[0] in thermo]
    if any(col in thermo for col in ("energy_eV_per_atom", "kinetic_energy_eV_per_atom", "total_energy_eV_per_atom")):
        panels.append(("energies", "Energy change (eV/atom)", True))

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.7 * len(panels)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, (col, ylabel, is_energy_panel) in zip(axes, panels):
        if not is_energy_panel:
            ax.plot(time, thermo[col])
            ax.set_ylabel(ylabel)
            continue
        for energy_col, label in [
            ("energy_eV_per_atom", "Potential"),
            ("kinetic_energy_eV_per_atom", "Kinetic"),
            ("total_energy_eV_per_atom", "Total"),
        ]:
            if energy_col in thermo:
                y = thermo[energy_col] - thermo[energy_col][0]
                ax.plot(time, y, label=label)
        ax.set_ylabel(ylabel)
        ax.legend()

    axes[-1].set_xlabel("Time (fs)")

    for ax in axes:
        ax.grid(alpha=0.25)

    out_dir = txt.parent / "plots"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{txt.stem}_summary.png"
    fig.suptitle(txt.parent.name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot thermo text output from an MLIP run folder.")
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    out = plot(parser.parse_args().folder)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
