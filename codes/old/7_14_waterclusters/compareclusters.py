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
from ase.io import read


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?)")
GRADIENT_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+:\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
DEFAULT_PLOT_MIN_EV = -8319.0
DEFAULT_PLOT_MAX_EV = -8314.0
FORCE_ELEMENT_COLORS = {
    "H": "#1f77b4",
    "O": "#d62728",
    "N": "#2ca02c",
    "S": "#9467bd",
}
DEFAULT_FORCE_OUTLIER_LABELS = 8


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
DEFAULT_ORCA_OUTPUT_DIR = REPO_ROOT / "outputscondensed" / "7_13a_expand"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputsfull" / "7_13a_orcaclusterssmall_comparison"
DEFAULT_CACHE_DIR = REPO_ROOT / "outputsfull" / ".cache"


def status(message: str) -> None:
    print(message, flush=True)


def resolve_cluster_dirs(cluster_root: Path) -> tuple[Path, Path]:
    """Return the XYZ cluster directory and legacy in-tree ORCA output directory.

    The comparison uses ``--orca-output-dir`` for DFT outputs by default, but
    this still accepts either the calculation root or its legacy ``expand``
    directory to locate sibling ``clusters/*.xyz`` geometries.
    """
    root = cluster_root.resolve()
    if root.name == "expand":
        out_dir = root
        xyz_dir = root.parent / "clusters"
    else:
        xyz_dir = root / "clusters"
        out_dir = root / "expand"
    return xyz_dir, out_dir


def completed_orca_output_paths(out_dir: Path) -> list[Path]:
    paths = sorted(out_dir.glob("*.out"))
    if not paths:
        raise FileNotFoundError(
            f"No ORCA .out files found in {out_dir}. "
            "Run the generated ORCA inputs first or pass --orca-output-dir."
        )
    return paths


def parse_orca_energy_hartree(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = FINAL_ENERGY_RE.findall(text)
    if not matches:
        raise ValueError(f"No FINAL SINGLE POINT ENERGY found in {path}")
    if "ORCA TERMINATED NORMALLY" not in text:
        raise ValueError(f"ORCA did not terminate normally in {path}")
    return float(matches[-1])


def parse_orca_gradient_hartree_per_bohr(path: Path, expected_symbols: list[str]) -> np.ndarray:
    """Return the final ORCA Cartesian gradient in Hartree/Bohr.

    ORCA prints gradients as dE/dR. Forces are therefore ``-gradient`` after
    converting Hartree/Bohr to eV/Angstrom.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    gradient_blocks: list[list[tuple[str, list[float]]]] = []
    for line_index, line in enumerate(lines):
        if line.strip() != "CARTESIAN GRADIENT":
            continue
        block: list[tuple[str, list[float]]] = []
        for gradient_line in lines[line_index + 1 :]:
            match = GRADIENT_LINE_RE.match(gradient_line)
            if match:
                _, symbol, gx, gy, gz = match.groups()
                block.append((symbol, [float(gx), float(gy), float(gz)]))
            elif block:
                break
        if block:
            gradient_blocks.append(block)

    if not gradient_blocks:
        raise ValueError(f"No CARTESIAN GRADIENT block found in {path}")

    block = gradient_blocks[-1]
    if len(block) != len(expected_symbols):
        raise ValueError(
            f"Gradient atom count mismatch in {path}: "
            f"ORCA has {len(block)}, XYZ has {len(expected_symbols)}"
        )
    gradient_symbols = [symbol for symbol, _ in block]
    if gradient_symbols != expected_symbols:
        raise ValueError(
            f"Gradient atom order mismatch in {path}: "
            f"ORCA symbols {gradient_symbols} differ from XYZ symbols {expected_symbols}"
        )
    return np.array([values for _, values in block], dtype=float)


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


def compare_clusters(
    cluster_root: Path,
    orca_output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    calc = build_calculator()
    records: list[dict[str, object]] = []
    force_records: list[dict[str, object]] = []
    xyz_dir, _ = resolve_cluster_dirs(cluster_root)

    for index, out_path in enumerate(completed_orca_output_paths(orca_output_dir), start=1):
        xyz_path = xyz_dir / f"{out_path.stem}.xyz"
        if not xyz_path.exists():
            raise FileNotFoundError(f"Missing cluster geometry for {out_path.name}: {xyz_path}")

        atoms = read(xyz_path)
        atoms.calc = calc
        mace_ev = float(atoms.get_potential_energy())
        mace_forces = np.array(atoms.get_forces(), dtype=float)

        dft_total_ev = parse_orca_energy_hartree(out_path) * HARTREE_TO_EV
        orca_gradient = parse_orca_gradient_hartree_per_bohr(out_path, atoms.get_chemical_symbols())
        orca_forces = -orca_gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        error_ev = mace_ev - dft_total_ev
        force_errors = mace_forces - orca_forces
        force_component_mae = float(np.mean(np.abs(force_errors)))
        force_component_rmse = float(np.sqrt(np.mean(force_errors**2)))
        force_vector_rmse = float(np.sqrt(np.mean(np.sum(force_errors**2, axis=1))))
        force_max_abs_error = float(np.max(np.abs(force_errors)))

        records.append(
            {
                "cluster_index": index,
                "xyz_file": xyz_path.name,
                "out_file": out_path.name,
                "natoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "mace_polar_cluster_energy_eV": mace_ev,
                "orca_total_energy_eV": dft_total_ev,
                "error_mace_minus_orca_total_eV": error_ev,
                "abs_error_eV": abs(error_ev),
                "force_component_mae_eV_A": force_component_mae,
                "force_component_rmse_eV_A": force_component_rmse,
                "force_vector_rmse_eV_A": force_vector_rmse,
                "force_max_abs_error_eV_A": force_max_abs_error,
            }
        )
        for atom_index, symbol in enumerate(atoms.get_chemical_symbols(), start=1):
            for component_index, component in enumerate(("x", "y", "z")):
                force_records.append(
                    {
                        "cluster_index": index,
                        "xyz_file": xyz_path.name,
                        "out_file": out_path.name,
                        "atom_index": atom_index,
                        "symbol": symbol,
                        "component": component,
                        "mace_force_eV_A": mace_forces[atom_index - 1, component_index],
                        "orca_force_eV_A": orca_forces[atom_index - 1, component_index],
                        "error_mace_minus_orca_eV_A": force_errors[atom_index - 1, component_index],
                        "abs_error_eV_A": abs(force_errors[atom_index - 1, component_index]),
                    }
                )
        status(
            f"{index:03d} {xyz_path.name}: "
            f"MACE={mace_ev:.6f} eV, ORCA-total={dft_total_ev:.6f} eV, "
            f"error={error_ev:.6f} eV, force-RMSE={force_component_rmse:.6f} eV/A"
        )

    return records, force_records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_comparison(
    path: Path,
    records: list[dict[str, object]],
    plot_min_ev: float | None = DEFAULT_PLOT_MIN_EV,
    plot_max_ev: float | None = DEFAULT_PLOT_MAX_EV,
) -> Path:
    dft = np.array([float(record["orca_total_energy_eV"]) for record in records])
    mace = np.array([float(record["mace_polar_cluster_energy_eV"]) for record in records])
    errors = mace - dft
    if plot_min_ev is not None and plot_max_ev is not None:
        mask = (
            (dft >= plot_min_ev)
            & (dft <= plot_max_ev)
            & (mace >= plot_min_ev)
            & (mace <= plot_max_ev)
        )
        dft_plot = dft[mask]
        mace_plot = mace[mask]
        if len(dft_plot) == 0:
            raise ValueError(
                f"No points fall inside the requested plot window "
                f"{plot_min_ev:g} to {plot_max_ev:g} eV on both axes."
            )
    else:
        dft_plot = dft
        mace_plot = mace
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_path = path.with_suffix(".svg")
        write_svg_plot(svg_path, dft_plot, mace_plot, errors, plot_min_ev, plot_max_ev)
        return svg_path

    if plot_min_ev is not None and plot_max_ev is not None:
        lo = plot_min_ev
        hi = plot_max_ev
    else:
        lo = min(dft_plot.min(), mace_plot.min())
        hi = max(dft_plot.max(), mace_plot.max())
        pad = 0.05 * max(hi - lo, 1.0)
        lo -= pad
        hi += pad

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(dft_plot, mace_plot, s=42, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, label="slope 1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DFT ORCA total energy (eV)")
    ax.set_ylabel("MACE-POLAR cluster energy (eV)")
    ax.set_title("Small cluster energy comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.text(
        0.04,
        0.96,
        f"MAE = {np.mean(np.abs(errors)):.4f} eV\nRMSE = {np.sqrt(np.mean(errors**2)):.4f} eV\nn = {len(dft_plot)}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def force_outlier_records(
    force_records: list[dict[str, object]],
    count: int = DEFAULT_FORCE_OUTLIER_LABELS,
) -> list[dict[str, object]]:
    return sorted(force_records, key=lambda record: float(record["abs_error_eV_A"]), reverse=True)[:count]


def force_point_label(record: dict[str, object]) -> str:
    return (
        f"c{int(record['cluster_index'])} "
        f"{record['symbol']}{int(record['atom_index'])}{record['component']}"
    )


def plot_force_comparison(
    path: Path,
    force_records: list[dict[str, object]],
    outlier_labels: int = DEFAULT_FORCE_OUTLIER_LABELS,
) -> Path:
    orca = np.array([float(record["orca_force_eV_A"]) for record in force_records])
    mace = np.array([float(record["mace_force_eV_A"]) for record in force_records])
    errors = mace - orca
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_path = path.with_suffix(".svg")
        write_svg_force_plot(svg_path, force_records, orca, mace, errors, outlier_labels)
        return svg_path

    lo = min(orca.min(), mace.min())
    hi = max(orca.max(), mace.max())
    pad = 0.05 * max(hi - lo, 1.0)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for symbol in sorted({str(record["symbol"]) for record in force_records}):
        mask = np.array([str(record["symbol"]) == symbol for record in force_records])
        ax.scatter(
            orca[mask],
            mace[mask],
            s=20,
            color=FORCE_ELEMENT_COLORS.get(symbol, "#7f7f7f"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.25,
            label=symbol,
        )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, label="slope 1")
    for record in force_outlier_records(force_records, outlier_labels):
        x = float(record["orca_force_eV_A"])
        y = float(record["mace_force_eV_A"])
        ax.annotate(
            force_point_label(record),
            xy=(x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("ORCA DFT force component (eV/A)")
    ax.set_ylabel("MACE-POLAR force component (eV/A)")
    ax.set_title("Small cluster force comparison")
    ax.grid(alpha=0.25)
    ax.legend(title="atom")
    ax.text(
        0.04,
        0.96,
        f"MAE = {np.mean(np.abs(errors)):.4f} eV/A\n"
        f"RMSE = {np.sqrt(np.mean(errors**2)):.4f} eV/A\n"
        f"n = {len(errors)} components",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_svg_plot(
    path: Path,
    dft: np.ndarray,
    mace: np.ndarray,
    errors: np.ndarray,
    plot_min_ev: float | None,
    plot_max_ev: float | None,
) -> None:
    width = 720
    height = 640
    margin_left = 95
    margin_right = 35
    margin_top = 45
    margin_bottom = 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    if plot_min_ev is not None and plot_max_ev is not None:
        lo = plot_min_ev
        hi = plot_max_ev
    else:
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
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial">DFT ORCA total energy (eV)</text>
<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-size="14" font-family="Arial">MACE-POLAR cluster energy (eV)</text>
<rect x="{margin_left + 15}" y="{margin_top + 12}" width="155" height="74" fill="white" stroke="#bbb"/>
<text x="{margin_left + 25}" y="{margin_top + 34}" font-size="13" font-family="Arial">MAE = {mae:.4f} eV</text>
<text x="{margin_left + 25}" y="{margin_top + 54}" font-size="13" font-family="Arial">RMSE = {rmse:.4f} eV</text>
<text x="{margin_left + 25}" y="{margin_top + 74}" font-size="13" font-family="Arial">n = {len(dft)}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_svg_force_plot(
    path: Path,
    force_records: list[dict[str, object]],
    orca: np.ndarray,
    mace: np.ndarray,
    errors: np.ndarray,
    outlier_labels: int,
) -> None:
    width = 720
    height = 640
    margin_left = 95
    margin_right = 35
    margin_top = 45
    margin_bottom = 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    lo = min(orca.min(), mace.min())
    hi = max(orca.max(), mace.max())
    pad = 0.05 * max(hi - lo, 1.0)
    lo -= pad
    hi += pad

    def sx(value: float) -> float:
        return margin_left + (value - lo) / (hi - lo) * plot_w

    def sy(value: float) -> float:
        return margin_top + (hi - value) / (hi - lo) * plot_h

    ticks = np.linspace(lo, hi, 6)
    points = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.0" '
        f'fill="{FORCE_ELEMENT_COLORS.get(str(record["symbol"]), "#7f7f7f")}" '
        f'fill-opacity="0.75" stroke="black" stroke-width="0.4">'
        f"<title>{force_point_label(record)}: ORCA {x:.6f} eV/A, MACE {y:.6f} eV/A</title></circle>"
        for record, x, y in zip(force_records, orca, mace)
    )
    outlier_labels_svg = "\n".join(
        f'<text x="{sx(float(record["orca_force_eV_A"])) + 5:.2f}" '
        f'y="{sy(float(record["mace_force_eV_A"])) - 5:.2f}" '
        f'font-size="10" font-family="Arial">{force_point_label(record)}</text>'
        for record in force_outlier_records(force_records, outlier_labels)
    )
    legend_symbols = sorted({str(record["symbol"]) for record in force_records})
    legend = "\n".join(
        f'<circle cx="{width - 120}" cy="{margin_top + 22 + i * 18}" r="4" '
        f'fill="{FORCE_ELEMENT_COLORS.get(symbol, "#7f7f7f")}" stroke="black" stroke-width="0.4"/>'
        f'<text x="{width - 108}" y="{margin_top + 26 + i * 18}" font-size="12" font-family="Arial">{symbol}</text>'
        for i, symbol in enumerate(legend_symbols)
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
<text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-family="Arial">Small cluster force comparison</text>
{grid}
<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black"/>
<line x1="{sx(lo):.2f}" y1="{sy(lo):.2f}" x2="{sx(hi):.2f}" y2="{sy(hi):.2f}" stroke="black" stroke-width="1.4"/>
{points}
{outlier_labels_svg}
{legend}
{x_labels}
{y_labels}
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial">ORCA DFT force component (eV/A)</text>
<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-size="14" font-family="Arial">MACE-POLAR force component (eV/A)</text>
<rect x="{margin_left + 15}" y="{margin_top + 12}" width="185" height="74" fill="white" stroke="#bbb"/>
<text x="{margin_left + 25}" y="{margin_top + 34}" font-size="13" font-family="Arial">MAE = {mae:.4f} eV/A</text>
<text x="{margin_left + 25}" y="{margin_top + 54}" font-size="13" font-family="Arial">RMSE = {rmse:.4f} eV/A</text>
<text x="{margin_left + 25}" y="{margin_top + 74}" font-size="13" font-family="Arial">n = {len(errors)} components</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--orca-output-dir", type=Path, default=DEFAULT_ORCA_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-min-ev", type=float, default=DEFAULT_PLOT_MIN_EV)
    parser.add_argument("--plot-max-ev", type=float, default=DEFAULT_PLOT_MAX_EV)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        records, force_records = compare_clusters(args.cluster_root, args.orca_output_dir)
    except ModuleNotFoundError as exc:
        if str(exc).startswith("MACE-Polar requires"):
            raise SystemExit(str(exc)) from None
        raise
    if not records:
        raise RuntimeError("No cluster comparisons were produced.")

    csv_path = args.output_dir / "mace_polar_vs_orca_small_clusters.csv"
    force_csv_path = args.output_dir / "mace_polar_vs_orca_force_components.csv"
    plot_path = args.output_dir / "mace_polar_vs_orca_small_clusters.png"
    force_plot_path = args.output_dir / "mace_polar_vs_orca_force_components.png"
    write_csv(csv_path, records)
    write_csv(force_csv_path, force_records)
    plot_path = plot_comparison(plot_path, records, args.plot_min_ev, args.plot_max_ev)
    force_plot_path = plot_force_comparison(force_plot_path, force_records)

    errors = np.array([float(record["error_mace_minus_orca_total_eV"]) for record in records])
    force_errors = np.array([float(record["error_mace_minus_orca_eV_A"]) for record in force_records])
    print(f"Compared clusters: {len(records)}")
    print(f"Mean error: {errors.mean():.6f} eV")
    print(f"MAE: {np.mean(np.abs(errors)):.6f} eV")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.6f} eV")
    print(f"Force component MAE: {np.mean(np.abs(force_errors)):.6f} eV/A")
    print(f"Force component RMSE: {np.sqrt(np.mean(force_errors**2)):.6f} eV/A")
    print(f"CSV: {csv_path}")
    print(f"Force CSV: {force_csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Force plot: {force_plot_path}")


if __name__ == "__main__":
    main()
