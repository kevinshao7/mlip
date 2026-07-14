#!/usr/bin/env python3
"""Compare MACE-POLAR cluster energies against ORCA DFT cluster energies."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
from ase import Atoms
from ase.io import read


HARTREE_TO_EV = 27.211386245988
FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?)")


# Match the MD scripts' CPU threading defaults unless the environment overrides them.
N_THREADS = os.environ.get("OMP_NUM_THREADS", "20")
os.environ.setdefault("OMP_NUM_THREADS", N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", N_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", N_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", N_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", N_THREADS)
os.environ.setdefault("TORCH_NUM_THREADS", N_THREADS)
os.environ.setdefault("OMP_PROC_BIND", "spread")
os.environ.setdefault("OMP_PLACES", "threads")

torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
torch.set_num_interop_threads(1)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CLUSTER_DIR = REPO_ROOT / "codes" / "7_13a_orcaclusterssmall"
DEFAULT_ATOMIZATION = SCRIPT_DIR / "atomizationenergies.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputsfull" / "7_13a_orcaclusterssmall_comparison"
DEFAULT_CACHE_DIR = REPO_ROOT / "outputsfull" / ".cache"


def status(message: str) -> None:
    print(message, flush=True)


def read_atomization_energies(path: Path) -> dict[str, float]:
    energies: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            symbol = row[0].strip()
            if symbol:
                energies[symbol] = float(row[1])
    if not energies:
        raise ValueError(f"No atomization energies found in {path}")
    return energies


def resolve_cluster_dirs(cluster_root: Path) -> tuple[Path, Path]:
    """Return the XYZ cluster directory and ORCA output directory.

    The validation inputs are split between sibling directories:
    ``clusters/*.xyz`` for geometries and ``expand/*.out`` for ORCA results.
    Accept either the calculation root or the ``expand`` directory itself.
    """
    root = cluster_root.resolve()
    if root.name == "expand":
        out_dir = root
        xyz_dir = root.parent / "clusters"
    else:
        xyz_dir = root / "clusters"
        out_dir = root / "expand"
    return xyz_dir, out_dir


def cluster_paths(cluster_root: Path) -> list[Path]:
    xyz_dir, _ = resolve_cluster_dirs(cluster_root)
    paths = sorted(xyz_dir.glob("*.xyz"))
    if not paths:
        raise FileNotFoundError(f"No cluster .xyz files found in {xyz_dir}")
    return paths


def parse_orca_energy_hartree(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = FINAL_ENERGY_RE.findall(text)
    if not matches:
        raise ValueError(f"No FINAL SINGLE POINT ENERGY found in {path}")
    if "ORCA TERMINATED NORMALLY" not in text:
        raise ValueError(f"ORCA did not terminate normally in {path}")
    return float(matches[-1])


def atomization_reference_energy(atoms: Atoms, atomization: dict[str, float]) -> float:
    total = 0.0
    for symbol in atoms.get_chemical_symbols():
        try:
            total += atomization[symbol]
        except KeyError as exc:
            raise KeyError(f"No DFT atomization/reference energy for atom {symbol!r}") from exc
    return total


def build_calculator():
    os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR))
    sys.path.insert(0, str(REPO_ROOT / "mace"))
    try:
        from mace.calculators import mace_polar
    except ModuleNotFoundError as exc:
        if exc.name == "graph_longrange":
            raise ModuleNotFoundError(
                "MACE-Polar requires graph_longrange/graph_electrostatics. "
                "Install that dependency or run in the environment used for MACE-Polar MLIP evaluation."
            ) from exc
        raise

    try:
        return mace_polar(
            model="polar-1-s",
            device=os.environ.get("MLIP_MACE_DEVICE", "cpu"),
            default_dtype="float32",
        )
    except ModuleNotFoundError as exc:
        if exc.name == "graph_longrange":
            raise ModuleNotFoundError(
                "MACE-Polar requires graph_longrange/graph_electrostatics. "
                "Install that dependency or run in the environment used for MACE-Polar MLIP evaluation."
            ) from exc
        raise


def compare_clusters(cluster_root: Path, atomization_path: Path) -> list[dict[str, object]]:
    atomization = read_atomization_energies(atomization_path)
    calc = build_calculator()
    records: list[dict[str, object]] = []
    _, out_dir = resolve_cluster_dirs(cluster_root)

    for index, xyz_path in enumerate(cluster_paths(cluster_root), start=1):
        out_path = out_dir / f"{xyz_path.stem}.out"
        if not out_path.exists():
            raise FileNotFoundError(f"Missing ORCA output for {xyz_path.name}: {out_path}")

        atoms = read(xyz_path)
        atoms.calc = calc
        mace_ev = float(atoms.get_potential_energy())

        dft_total_ev = parse_orca_energy_hartree(out_path) * HARTREE_TO_EV
        atomic_reference_ev = atomization_reference_energy(atoms, atomization)
        dft_relative_ev = dft_total_ev - atomic_reference_ev
        mace_relative_ev = mace_ev - atomic_reference_ev
        error_ev = mace_relative_ev - dft_relative_ev

        records.append(
            {
                "cluster_index": index,
                "xyz_file": xyz_path.name,
                "out_file": out_path.name,
                "natoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "mace_polar_total_energy_eV": mace_ev,
                "mace_polar_relative_energy_eV": mace_relative_ev,
                "dft_total_energy_eV": dft_total_ev,
                "dft_atomic_reference_eV": atomic_reference_ev,
                "dft_relative_energy_eV": dft_relative_ev,
                "error_eV": error_ev,
                "abs_error_eV": abs(error_ev),
            }
        )
        status(
            f"{index:03d} {xyz_path.name}: "
            f"MACE-relative={mace_relative_ev:.6f} eV, DFT-relative={dft_relative_ev:.6f} eV, "
            f"error={error_ev:.6f} eV"
        )

    return records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_comparison(path: Path, records: list[dict[str, object]]) -> Path:
    dft = np.array([float(record["dft_relative_energy_eV"]) for record in records])
    mace = np.array([float(record["mace_polar_relative_energy_eV"]) for record in records])
    errors = mace - dft
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_path = path.with_suffix(".svg")
        write_svg_plot(svg_path, dft, mace, errors)
        return svg_path

    lo = min(dft.min(), mace.min())
    hi = max(dft.max(), mace.max())
    pad = 0.05 * max(hi - lo, 1.0)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(dft, mace, s=42, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1.2, label="slope 1")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("DFT ORCA energy minus atomic references (eV)")
    ax.set_ylabel("MACE-POLAR energy minus atomic references (eV)")
    ax.set_title("Small cluster energy comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.text(
        0.04,
        0.96,
        f"MAE = {np.mean(np.abs(errors)):.4f} eV\nRMSE = {np.sqrt(np.mean(errors**2)):.4f} eV",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_svg_plot(path: Path, dft: np.ndarray, mace: np.ndarray, errors: np.ndarray) -> None:
    width = 720
    height = 640
    margin_left = 95
    margin_right = 35
    margin_top = 45
    margin_bottom = 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    lo = min(dft.min(), mace.min())
    hi = max(dft.max(), mace.max())
    pad = 0.05 * max(hi - lo, 1.0)
    lo -= pad
    hi += pad

    def sx(value: float) -> float:
        return margin_left + (value - lo) / (hi - lo) * plot_w

    def sy(value: float) -> float:
        return margin_top + (hi - value) / (hi - lo) * plot_h

    ticks = np.linspace(lo, hi, 6)
    points = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="#1f77b4" '
        f'stroke="black" stroke-width="0.7"><title>DFT {x:.6f} eV, MACE {y:.6f} eV</title></circle>'
        for x, y in zip(dft, mace)
    )
    grid = "\n".join(
        f'<line x1="{sx(t):.2f}" y1="{margin_top}" x2="{sx(t):.2f}" y2="{margin_top + plot_h}" stroke="#ddd"/>'
        f'<line x1="{margin_left}" y1="{sy(t):.2f}" x2="{margin_left + plot_w}" y2="{sy(t):.2f}" stroke="#ddd"/>'
        for t in ticks
    )
    x_labels = "\n".join(
        f'<text x="{sx(t):.2f}" y="{height - 55}" text-anchor="middle" font-size="12">{t:.1f}</text>'
        for t in ticks
    )
    y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{sy(t) + 4:.2f}" text-anchor="end" font-size="12">{t:.1f}</text>'
        for t in ticks
    )
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-family="Arial">Small cluster energy comparison</text>
{grid}
<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black"/>
<line x1="{sx(lo):.2f}" y1="{sy(lo):.2f}" x2="{sx(hi):.2f}" y2="{sy(hi):.2f}" stroke="black" stroke-width="1.4"/>
{points}
{x_labels}
{y_labels}
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial">DFT ORCA energy minus atomic references (eV)</text>
<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-size="14" font-family="Arial">MACE-POLAR energy minus atomic references (eV)</text>
<rect x="{margin_left + 15}" y="{margin_top + 12}" width="155" height="54" fill="white" stroke="#bbb"/>
<text x="{margin_left + 25}" y="{margin_top + 34}" font-size="13" font-family="Arial">MAE = {mae:.4f} eV</text>
<text x="{margin_left + 25}" y="{margin_top + 54}" font-size="13" font-family="Arial">RMSE = {rmse:.4f} eV</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--atomization", type=Path, default=DEFAULT_ATOMIZATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        records = compare_clusters(args.cluster_root, args.atomization)
    except ModuleNotFoundError as exc:
        if str(exc).startswith("MACE-Polar requires"):
            raise SystemExit(str(exc)) from None
        raise
    if not records:
        raise RuntimeError("No cluster comparisons were produced.")

    csv_path = args.output_dir / "mace_polar_vs_orca_small_clusters.csv"
    plot_path = args.output_dir / "mace_polar_vs_orca_small_clusters.png"
    write_csv(csv_path, records)
    plot_path = plot_comparison(plot_path, records)

    errors = np.array([float(record["error_eV"]) for record in records])
    print(f"Compared clusters: {len(records)}")
    print(f"Mean error: {errors.mean():.6f} eV")
    print(f"MAE: {np.mean(np.abs(errors)):.6f} eV")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.6f} eV")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
