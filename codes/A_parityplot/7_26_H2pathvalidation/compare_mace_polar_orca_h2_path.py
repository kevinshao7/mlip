#!/usr/bin/env python3
"""Compare MACE-POLAR and ORCA DFT along the H2-formation validation path."""

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


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
ORCA_NORMAL_TERMINATION = "ORCA TERMINATED NORMALLY"
FRAME_RE = re.compile(r"_frame_(\d+)\.(?:inp|out)$")
XYZ_BLOCK_HEADER_RE = re.compile(r"^\*xyz\s+(-?\d+)\s+(\d+)\s*$", re.IGNORECASE)
XYZ_BLOCK_ATOM_RE = re.compile(
    r"^\s*([A-Za-z]+)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
GRADIENT_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+:\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
VALID_FRAMES = tuple(range(1, 7)) + tuple(range(11, 16))
FORCE_ELEMENT_COLORS = {
    "H": "#1f77b4",
    "O": "#d62728",
    "N": "#2ca02c",
    "S": "#9467bd",
}


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
REPO_ROOT = SCRIPT_DIR.parents[1] if len(SCRIPT_DIR.parents) > 1 else Path("/home/kevinsh/mlip")
DEFAULT_DATA_DIR = REPO_ROOT / "codes" / "7_25_h2formationorca"
DEFAULT_ATOMIC_REFERENCE_PATH = REPO_ROOT / "codes" / "7_7b_clustervalidation" / "atomizationenergies.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputsfull" / "7_26_H2pathvalidation"
DEFAULT_CACHE_DIR = REPO_ROOT / "outputsfull" / ".cache"


def status(message: str) -> None:
    print(message, flush=True)


def load_atomic_reference_energies(path: Path) -> dict[str, float]:
    references: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            references[row[0].strip()] = float(row[1].strip())
    if not references:
        raise ValueError(f"No atomic reference energies found in {path}")
    return references


def atomic_reference_sum_ev(symbols: list[str], references: dict[str, float]) -> float:
    missing = sorted({symbol for symbol in symbols if symbol not in references})
    if missing:
        raise KeyError(f"Missing atomic reference energies for: {', '.join(missing)}")
    return float(sum(references[symbol] for symbol in symbols))


def frame_index_from_path(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse frame index from {path.name}")
    return int(match.group(1))


def parse_orca_energy_hartree(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = FINAL_ENERGY_RE.findall(text)
    if not matches:
        raise ValueError(f"No FINAL SINGLE POINT ENERGY found in {path}")
    if ORCA_NORMAL_TERMINATION not in text:
        raise ValueError(f"ORCA did not terminate normally in {path}")
    return float(matches[-1])


def parse_orca_gradient_hartree_per_bohr(path: Path, expected_symbols: list[str]) -> np.ndarray:
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
            f"Gradient atom count mismatch in {path}: ORCA has {len(block)}, XYZ has {len(expected_symbols)}"
        )
    gradient_symbols = [symbol for symbol, _ in block]
    if gradient_symbols != expected_symbols:
        raise ValueError(
            f"Gradient atom order mismatch in {path}: ORCA symbols {gradient_symbols} differ from XYZ symbols {expected_symbols}"
        )
    return np.array([values for _, values in block], dtype=float)


def atoms_from_orca_input(path: Path) -> Atoms:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    xyz_start = None
    for line_index, line in enumerate(lines):
        if XYZ_BLOCK_HEADER_RE.match(line.strip()):
            xyz_start = line_index + 1
            break
    if xyz_start is None:
        raise ValueError(f"No *xyz block found in {path}")

    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[xyz_start:]:
        stripped = line.strip()
        if stripped == "*":
            break
        match = XYZ_BLOCK_ATOM_RE.match(line)
        if not match:
            raise ValueError(f"Malformed XYZ line in {path}: {line}")
        symbol, x, y, z = match.groups()
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    if not symbols:
        raise ValueError(f"Empty *xyz block in {path}")
    return Atoms(symbols=symbols, positions=np.array(positions, dtype=float))


def completed_orca_output_paths(data_dir: Path) -> list[Path]:
    selected = [data_dir / f"r09_hot_w_h2formation_frame_{frame:03d}.out" for frame in VALID_FRAMES]
    completed: list[Path] = []
    for path in selected:
        if not path.exists():
            raise FileNotFoundError(f"Missing required ORCA output for validation frame {frame_index_from_path(path)}: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        if ORCA_NORMAL_TERMINATION not in text:
            raise RuntimeError(f"ORCA did not terminate normally in {path}")
        if not FINAL_ENERGY_RE.search(text):
            raise RuntimeError(f"Missing FINAL SINGLE POINT ENERGY in {path}")
        if "CARTESIAN GRADIENT" not in text:
            raise RuntimeError(f"Missing CARTESIAN GRADIENT in {path}")
        completed.append(path)
    return completed


def build_calculator():
    os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR))
    sys.path.insert(0, str(REPO_ROOT / "mace"))
    device = os.environ.get("MLIP_MACE_DEVICE", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "MLIP_MACE_DEVICE requests CUDA, but torch.cuda.is_available() is False. "
            "Set MLIP_MACE_DEVICE=cpu to run on CPU."
        )
    status(f"Using MACE-POLAR device: {device}")
    try:
        from mace.calculators import mace_polar
    except ModuleNotFoundError as exc:
        if exc.name == "graph_longrange":
            raise ModuleNotFoundError(
                "MACE-Polar requires graph_longrange/graph_electrostatics. "
                "Install that dependency or run in the environment used for MACE-Polar MLIP evaluation."
            ) from exc
        raise
    return mace_polar(
        model="polar-1-s",
        device=device,
        default_dtype="float32",
    )


def compare_path(
    data_dir: Path,
    atomic_reference_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    calc = build_calculator()
    atomic_references = load_atomic_reference_energies(atomic_reference_path)
    energy_records: list[dict[str, object]] = []
    force_records: list[dict[str, object]] = []

    for out_path in completed_orca_output_paths(data_dir):
        frame = frame_index_from_path(out_path)
        inp_path = data_dir / f"r09_hot_w_h2formation_frame_{frame:03d}.inp"
        if not inp_path.exists():
            raise FileNotFoundError(f"Missing ORCA input for frame {frame}: {inp_path}")

        atoms = atoms_from_orca_input(inp_path)
        atoms.calc = calc
        mace_total_ev = float(atoms.get_potential_energy())
        mace_forces = np.array(atoms.get_forces(), dtype=float)

        reference_sum_ev = atomic_reference_sum_ev(atoms.get_chemical_symbols(), atomic_references)
        dft_total_ev = parse_orca_energy_hartree(out_path) * HARTREE_TO_EV
        mace_referenced_ev = mace_total_ev - reference_sum_ev
        dft_referenced_ev = dft_total_ev - reference_sum_ev
        energy_error_ev = mace_referenced_ev - dft_referenced_ev

        orca_gradient = parse_orca_gradient_hartree_per_bohr(out_path, atoms.get_chemical_symbols())
        orca_forces = -orca_gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        force_errors = mace_forces - orca_forces

        energy_records.append(
            {
                "frame": frame,
                "inp_file": inp_path.name,
                "out_file": out_path.name,
                "natoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "atomic_reference_sum_eV": reference_sum_ev,
                "mace_polar_total_energy_eV": mace_total_ev,
                "mace_polar_referenced_energy_eV": mace_referenced_ev,
                "orca_total_energy_eV": dft_total_ev,
                "orca_referenced_energy_eV": dft_referenced_ev,
                "error_mace_minus_orca_referenced_eV": energy_error_ev,
                "abs_error_eV": abs(energy_error_ev),
                "force_component_mae_eV_A": float(np.mean(np.abs(force_errors))),
                "force_component_rmse_eV_A": float(np.sqrt(np.mean(force_errors**2))),
                "force_vector_rmse_eV_A": float(np.sqrt(np.mean(np.sum(force_errors**2, axis=1)))),
                "force_max_abs_error_eV_A": float(np.max(np.abs(force_errors))),
            }
        )

        for atom_index, symbol in enumerate(atoms.get_chemical_symbols(), start=1):
            for component_index, component in enumerate(("x", "y", "z")):
                force_records.append(
                    {
                        "frame": frame,
                        "inp_file": inp_path.name,
                        "out_file": out_path.name,
                        "atom_index": atom_index,
                        "symbol": symbol,
                        "component": component,
                        "mace_force_eV_A": float(mace_forces[atom_index - 1, component_index]),
                        "orca_force_eV_A": float(orca_forces[atom_index - 1, component_index]),
                        "error_mace_minus_orca_eV_A": float(force_errors[atom_index - 1, component_index]),
                        "abs_error_eV_A": float(abs(force_errors[atom_index - 1, component_index])),
                    }
                )

        status(
            f"frame {frame:03d}: "
            f"MACE-ref={mace_referenced_ev:.6f} eV, ORCA-ref={dft_referenced_ev:.6f} eV, "
            f"energy error={energy_error_ev:.6f} eV, "
            f"force RMSE={np.sqrt(np.mean(force_errors**2)):.6f} eV/A"
        )

    return energy_records, force_records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_energy_parity(path: Path, energy_records: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    frames = np.array([int(record["frame"]) for record in energy_records])
    orca = np.array([float(record["orca_referenced_energy_eV"]) for record in energy_records])
    mace = np.array([float(record["mace_polar_referenced_energy_eV"]) for record in energy_records])
    natoms = np.array([int(record["natoms"]) for record in energy_records], dtype=float)
    errors_mev_per_atom = 1000.0 * (mace - orca) / natoms
    correlation = float(np.corrcoef(orca, mace)[0, 1]) if len(energy_records) > 1 else float("nan")

    lo = min(orca.min(), mace.min())
    hi = max(orca.max(), mace.max())
    pad = 0.05 * max(hi - lo, 1.0)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(orca, mace, s=48, color="#1f77b4", edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2)
    for frame, x, y in zip(frames, orca, mace):
        ax.annotate(f"{frame:03d}", xy=(x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("ORCA referenced energy (eV)")
    ax.set_ylabel("MACE-POLAR referenced energy (eV)")
    ax.set_title("H2-formation path energy parity")
    ax.grid(alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f"MAE = {np.mean(np.abs(errors_mev_per_atom)):.2f} meV/atom\n"
        f"RMSE = {np.sqrt(np.mean(errors_mev_per_atom**2)):.2f} meV/atom\n"
        f"r = {correlation:.4f}\n"
        f"n = {len(energy_records)} frames",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_force_parity(path: Path, force_records: list[dict[str, object]], title: str) -> Path:
    import matplotlib.pyplot as plt

    orca = np.array([float(record["orca_force_eV_A"]) for record in force_records])
    mace = np.array([float(record["mace_force_eV_A"]) for record in force_records])
    errors = mace - orca
    frames = np.array([int(record["frame"]) for record in force_records])
    correlation = float(np.corrcoef(orca, mace)[0, 1]) if len(force_records) > 1 else float("nan")

    lo = min(orca.min(), mace.min())
    hi = max(orca.max(), mace.max())
    pad = 0.05 * max(hi - lo, 1.0)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    frame_values = sorted(set(int(frame) for frame in frames))
    cmap = plt.get_cmap("tab20")
    for color_index, frame in enumerate(frame_values):
        mask = frames == frame
        ax.scatter(
            orca[mask],
            mace[mask],
            s=18,
            color=cmap(color_index % cmap.N),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.2,
            label=f"{frame:03d}",
        )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("ORCA DFT force component (eV/A)")
    ax.set_ylabel("MACE-POLAR force component (eV/A)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(title="frame", ncol=2, fontsize=8)
    ax.text(
        0.04,
        0.96,
        f"MAE = {np.mean(np.abs(errors)):.4f} eV/A\n"
        f"RMSE = {np.sqrt(np.mean(errors**2)):.4f} eV/A\n"
        f"r = {correlation:.4f}\n"
        f"n = {len(errors) // 3} atoms ({len(errors)} components)",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_force_parity_by_frame(output_dir: Path, force_records: list[dict[str, object]]) -> list[Path]:
    import matplotlib.pyplot as plt

    per_frame_dir = output_dir / "force_parity_by_frame"
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    frames = sorted({int(record["frame"]) for record in force_records})
    for frame in frames:
        frame_records = [record for record in force_records if int(record["frame"]) == frame]
        orca = np.array([float(record["orca_force_eV_A"]) for record in frame_records])
        mace = np.array([float(record["mace_force_eV_A"]) for record in frame_records])
        errors = mace - orca
        correlation = float(np.corrcoef(orca, mace)[0, 1]) if len(frame_records) > 1 else float("nan")

        lo = min(orca.min(), mace.min())
        hi = max(orca.max(), mace.max())
        pad = 0.05 * max(hi - lo, 1.0)
        lo -= pad
        hi += pad

        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        for symbol in sorted({str(record["symbol"]) for record in frame_records}):
            mask = np.array([str(record["symbol"]) == symbol for record in frame_records])
            ax.scatter(
                orca[mask],
                mace[mask],
                s=28,
                color=FORCE_ELEMENT_COLORS.get(symbol, "#7f7f7f"),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.25,
                label=symbol,
            )
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("ORCA DFT force component (eV/A)")
        ax.set_ylabel("MACE-POLAR force component (eV/A)")
        ax.set_title(f"H2-formation force parity, frame {frame:03d}")
        ax.grid(alpha=0.25)
        ax.legend(title="atom")
        ax.text(
            0.04,
            0.96,
            f"MAE = {np.mean(np.abs(errors)):.4f} eV/A\n"
            f"RMSE = {np.sqrt(np.mean(errors**2)):.4f} eV/A\n"
            f"r = {correlation:.4f}\n"
            f"n = {len(errors) // 3} atoms ({len(errors)} components)",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        fig.tight_layout()
        plot_path = per_frame_dir / f"force_parity_frame_{frame:03d}.png"
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)
        written.append(plot_path)
    return written


def plot_pathway_summary(path: Path, energy_records: list[dict[str, object]]) -> Path:
    import matplotlib.pyplot as plt

    frames = np.array([int(record["frame"]) for record in energy_records], dtype=int)
    dft_referenced = np.array([float(record["orca_referenced_energy_eV"]) for record in energy_records])
    mace_referenced = np.array([float(record["mace_polar_referenced_energy_eV"]) for record in energy_records])
    force_rmse = np.array([float(record["force_component_rmse_eV_A"]) for record in energy_records])
    natoms = np.array([int(record["natoms"]) for record in energy_records], dtype=float)
    energy_error_mev_per_atom = 1000.0 * (mace_referenced - dft_referenced) / natoms

    fig, (ax_energy, ax_force) = plt.subplots(2, 1, figsize=(7.0, 7.4), sharex=True)

    ax_energy.plot(frames, dft_referenced, marker="o", linewidth=1.6, color="#222222", label="ORCA DFT")
    ax_energy.plot(frames, mace_referenced, marker="s", linewidth=1.6, color="#1f77b4", label="MACE-POLAR")
    for frame, error in zip(frames, energy_error_mev_per_atom):
        ax_energy.annotate(f"{error:+.1f}", xy=(frame, mace_referenced[list(frames).index(frame)]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    ax_energy.set_ylabel("Referenced energy (eV)")
    ax_energy.set_title("H2-formation pathway validation")
    ax_energy.grid(alpha=0.25)
    ax_energy.legend()
    ax_energy.text(
        0.02,
        0.96,
        f"Energy MAE = {np.mean(np.abs(energy_error_mev_per_atom)):.2f} meV/atom\n"
        f"Energy RMSE = {np.sqrt(np.mean(energy_error_mev_per_atom**2)):.2f} meV/atom",
        transform=ax_energy.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    ax_force.plot(frames, force_rmse, marker="D", linewidth=1.6, color="#d62728")
    ax_force.set_xlabel("Frame index")
    ax_force.set_ylabel("Force component RMSE (eV/A)")
    ax_force.grid(alpha=0.25)
    ax_force.text(
        0.02,
        0.96,
        f"Mean force RMSE = {np.mean(force_rmse):.4f} eV/A\n"
        f"Max force RMSE = {np.max(force_rmse):.4f} eV/A",
        transform=ax_force.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax_force.set_xticks(frames)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--atomic-reference-path", type=Path, default=DEFAULT_ATOMIC_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        energy_records, force_records = compare_path(args.data_dir, args.atomic_reference_path)
    except ModuleNotFoundError as exc:
        if str(exc).startswith("MACE-Polar requires"):
            raise SystemExit(str(exc)) from None
        raise

    energy_csv = args.output_dir / "mace_polar_vs_orca_h2_path_energies.csv"
    force_csv = args.output_dir / "mace_polar_vs_orca_h2_path_force_components.csv"
    energy_plot = args.output_dir / "mace_polar_vs_orca_h2_path_energy_parity.png"
    force_plot = args.output_dir / "mace_polar_vs_orca_h2_path_force_parity.png"
    pathway_plot = args.output_dir / "mace_polar_vs_orca_h2_path_pathway_summary.png"

    write_csv(energy_csv, energy_records)
    write_csv(force_csv, force_records)
    plot_energy_parity(energy_plot, energy_records)
    plot_force_parity(force_plot, force_records, "H2-formation path force parity")
    plot_pathway_summary(pathway_plot, energy_records)
    per_frame_paths = plot_force_parity_by_frame(args.output_dir, force_records)

    energy_errors = np.array([float(record["error_mace_minus_orca_referenced_eV"]) for record in energy_records])
    force_errors = np.array([float(record["error_mace_minus_orca_eV_A"]) for record in force_records])
    frame_labels = ", ".join(f"{int(record['frame']):03d}" for record in energy_records)
    print(f"Compared frames: {len(energy_records)}")
    print(f"Frames: {frame_labels}")
    print(f"Referenced energy MAE: {np.mean(np.abs(energy_errors)):.6f} eV")
    print(f"Referenced energy RMSE: {np.sqrt(np.mean(energy_errors**2)):.6f} eV")
    print(f"Force component MAE: {np.mean(np.abs(force_errors)):.6f} eV/A")
    print(f"Force component RMSE: {np.sqrt(np.mean(force_errors**2)):.6f} eV/A")
    print(f"Energy CSV: {energy_csv}")
    print(f"Force CSV: {force_csv}")
    print(f"Energy parity plot: {energy_plot}")
    print(f"Force parity plot: {force_plot}")
    print(f"Pathway summary plot: {pathway_plot}")
    print(f"Per-frame force parity plots: {len(per_frame_paths)} files in {args.output_dir / 'force_parity_by_frame'}")


if __name__ == "__main__":
    main()
