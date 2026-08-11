#!/usr/bin/env python3
"""Plot MLIP-vs-DFT error histograms for parity-plot clusters."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
GRADIENT_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]+)\s+:\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
DFT_FRAME_RE = re.compile(r"_(\d+)\.out$")
ORCA_NORMAL_TERMINATION = "ORCA TERMINATED NORMALLY"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DFT_DIR = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_5_bluehiveDFT"
DEFAULT_MLIP_ROOT = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_6_mlippred"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_7_histogram"
DEFAULT_ATOMIC_REFERENCE = REPO_ROOT / "codes" / "7_7b_clustervalidation" / "atomizationenergies.txt"
MODEL_LABELS = {
    "polar1s": "mace_polar_1s",
    "polar1m": "mace_polar_1m",
    "off": "mace_off_medium",
}
ATOM_CLASS_COLORS = {
    "target isolated H": "#d62728",
    "nearest atom": "#1f77b4",
    "all other atoms": "#7f7f7f",
}
ENERGY_COLOR = "#2ca02c"
EPS = 1.0e-12


def status(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MLIP prediction CSVs with ORCA DFT outputs and plot four error histograms."
    )
    parser.add_argument("--model", choices=tuple(MODEL_LABELS), default="polar1s")
    parser.add_argument("--dft-dir", type=Path, default=DEFAULT_DFT_DIR)
    parser.add_argument("--mlip-root", type=Path, default=DEFAULT_MLIP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--atomic-reference",
        type=Path,
        default=DEFAULT_ATOMIC_REFERENCE,
        help="CSV with Atom and DFT Atomization energies columns. Use --no-reference for raw total energies.",
    )
    parser.add_argument("--no-reference", action="store_true", help="Compare raw total energies.")
    parser.add_argument("--force-bins", type=int, default=60)
    parser.add_argument("--energy-bins", type=int, default=40)
    parser.add_argument("--fractional-eps", type=float, default=EPS)
    parser.add_argument(
        "--target-position",
        type=parse_vector3,
        default=(12.0, 12.0, 12.0),
        help="Position of the centered isolated H atom in Angstrom. Default: 12,12,12.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_vector3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("must have three comma-separated values, e.g. 12,12,12")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    label = MODEL_LABELS[args.model]
    model_dir = args.mlip_root / args.model
    summary_matches = sorted(model_dir.glob(f"*_{label}_singlepoints.csv"))
    force_matches = sorted(model_dir.glob(f"*_{label}_forces.csv"))
    if len(summary_matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {label} singlepoints CSV in {model_dir}, found {len(summary_matches)}")
    if len(force_matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {label} forces CSV in {model_dir}, found {len(force_matches)}")
    return summary_matches[0], force_matches[0]


def expected_model_globs(args: argparse.Namespace) -> tuple[Path, str, str]:
    label = MODEL_LABELS[args.model]
    model_dir = args.mlip_root / args.model
    return model_dir, f"*_{label}_singlepoints.csv", f"*_{label}_forces.csv"


def frame_from_dft_path(path: Path) -> int:
    match = DFT_FRAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse frame from DFT output name: {path.name}")
    return int(match.group(1))


def parse_dft_output(path: Path) -> tuple[float, list[str], np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if ORCA_NORMAL_TERMINATION not in text:
        raise RuntimeError(f"ORCA did not terminate normally: {path}")

    energy_matches = FINAL_ENERGY_RE.findall(text)
    if not energy_matches:
        raise ValueError(f"No FINAL SINGLE POINT ENERGY found in {path}")
    energy_ev = float(energy_matches[-1]) * HARTREE_TO_EV

    lines = text.splitlines()
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
    symbols = [symbol for symbol, _ in block]
    gradient_hartree_bohr = np.array([values for _, values in block], dtype=float)
    forces_ev_a = -gradient_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    return energy_ev, symbols, forces_ev_a


def load_dft_outputs(dft_dir: Path) -> dict[int, dict[str, object]]:
    outputs: dict[int, dict[str, object]] = {}
    for path in sorted(dft_dir.glob("*.out")):
        frame = frame_from_dft_path(path)
        try:
            energy_ev, symbols, forces_ev_a = parse_dft_output(path)
        except Exception as exc:
            status(f"Skipping DFT frame {frame}: {exc}")
            continue
        outputs[frame] = {"energy_ev": energy_ev, "symbols": symbols, "forces": forces_ev_a}
    if not outputs:
        raise RuntimeError(f"No complete ORCA outputs found in {dft_dir}")
    return outputs


def load_mlip_summary(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {int(row["frame"]): row for row in reader if row.get("status") == "ok"}
    if not rows:
        raise RuntimeError(f"No successful MLIP frames found in {path}")
    return rows


def load_mlip_forces(path: Path) -> dict[int, list[dict[str, str]]]:
    frames: dict[int, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frames.setdefault(int(row["frame"]), []).append(row)
    for rows in frames.values():
        rows.sort(key=lambda row: int(row["atom_index"]))
    if not frames:
        raise RuntimeError(f"No MLIP force rows found in {path}")
    return frames


def load_atomic_references(path: Path) -> dict[str, float]:
    references: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                references[row[0].strip()] = float(row[1])
    if not references:
        raise RuntimeError(f"No atomic references loaded from {path}")
    return references


def reference_sum(symbols: list[str], references: dict[str, float]) -> float:
    missing = sorted({symbol for symbol in symbols if symbol not in references})
    if missing:
        raise KeyError(f"Missing atomic reference energies for: {', '.join(missing)}")
    return float(sum(references[symbol] for symbol in symbols))


def mlip_force_array(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    symbols = [row["symbol"] for row in rows]
    positions = np.array([[float(row["x_A"]), float(row["y_A"]), float(row["z_A"])] for row in rows], dtype=float)
    forces = np.array(
        [[float(row["force_x_eV_A"]), float(row["force_y_eV_A"]), float(row["force_z_eV_A"])] for row in rows],
        dtype=float,
    )
    return symbols, positions, forces


def atom_classes(
    symbols: list[str],
    positions: np.ndarray,
    target_position: tuple[float, float, float],
) -> list[str]:
    target_point = np.asarray(target_position, dtype=float)
    hydrogen_indices = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    if not hydrogen_indices:
        raise ValueError("No H atoms found; cannot identify target isolated H")

    center_distances = np.array([np.linalg.norm(positions[index] - target_point) for index in hydrogen_indices])
    target_index = hydrogen_indices[int(np.argmin(center_distances))]
    if float(np.min(center_distances)) > 1.0e-5:
        raise ValueError(
            f"No H atom found at target position {target_position}; closest H index {target_index} "
            f"is {float(np.min(center_distances)):.6g} A away"
        )

    distances = np.linalg.norm(positions - positions[target_index], axis=1)
    distances[target_index] = math.inf
    nearest_index = int(np.argmin(distances))
    classes = ["all other atoms"] * len(symbols)
    classes[target_index] = "target isolated H"
    classes[nearest_index] = "nearest atom"
    return classes


def collect_errors(
    dft: dict[int, dict[str, object]],
    mlip_summary: dict[int, dict[str, str]],
    mlip_forces: dict[int, list[dict[str, str]]],
    references: dict[str, float] | None,
    fractional_eps: float,
    target_position: tuple[float, float, float],
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[float], list[float], list[int]]:
    force_abs_by_class = {label: [] for label in ATOM_CLASS_COLORS}
    force_frac_by_class = {label: [] for label in ATOM_CLASS_COLORS}
    energy_abs: list[float] = []
    energy_frac: list[float] = []
    matched_frames: list[int] = []

    common_frames = sorted(set(dft) & set(mlip_summary) & set(mlip_forces))
    for frame in common_frames:
        dft_record = dft[frame]
        dft_symbols = list(dft_record["symbols"])  # type: ignore[arg-type]
        dft_forces = np.asarray(dft_record["forces"], dtype=float)
        mlip_symbols, positions, mlip_forces_array = mlip_force_array(mlip_forces[frame])
        if mlip_symbols != dft_symbols:
            raise ValueError(f"Atom order mismatch for frame {frame}: MLIP {mlip_symbols} vs DFT {dft_symbols}")
        if mlip_forces_array.shape != dft_forces.shape:
            raise ValueError(f"Force shape mismatch for frame {frame}: MLIP {mlip_forces_array.shape}, DFT {dft_forces.shape}")

        classes = atom_classes(mlip_symbols, positions, target_position)
        force_errors = np.linalg.norm(mlip_forces_array - dft_forces, axis=1)
        dft_force_magnitudes = np.linalg.norm(dft_forces, axis=1)
        force_fractional = force_errors / np.maximum(dft_force_magnitudes, fractional_eps)
        for atom_class, absolute_error, fractional_error in zip(classes, force_errors, force_fractional):
            force_abs_by_class[atom_class].append(float(absolute_error))
            force_frac_by_class[atom_class].append(float(fractional_error))

        dft_energy = float(dft_record["energy_ev"])
        mlip_energy = float(mlip_summary[frame]["energy_eV"])
        if references is not None:
            ref_sum = reference_sum(dft_symbols, references)
            dft_energy -= ref_sum
            mlip_energy -= ref_sum
        absolute_energy_error = abs(mlip_energy - dft_energy)
        fractional_energy_error = absolute_energy_error / max(abs(dft_energy), fractional_eps)
        energy_abs.append(float(absolute_energy_error))
        energy_frac.append(float(fractional_energy_error))
        matched_frames.append(frame)

    if not matched_frames:
        raise RuntimeError("No common complete frames found between DFT outputs and MLIP predictions")
    return force_abs_by_class, force_frac_by_class, energy_abs, energy_frac, matched_frames


def finite_values(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def plot_force_histogram(
    values_by_class: dict[str, list[float]],
    xlabel: str,
    title: str,
    output_path: Path,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    plotted = False
    for label, color in ATOM_CLASS_COLORS.items():
        values = finite_values(values_by_class[label])
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.48, color=color, edgecolor=color, label=label)
        plotted = True
    if not plotted:
        raise RuntimeError(f"No finite values to plot for {output_path}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of atoms")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, color="0.88", linewidth=0.7)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_energy_histogram(values: list[float], xlabel: str, title: str, output_path: Path, bins: int) -> None:
    finite = finite_values(values)
    if finite.size == 0:
        raise RuntimeError(f"No finite values to plot for {output_path}")
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.hist(finite, bins=bins, histtype="stepfilled", alpha=0.68, color=ENERGY_COLOR, edgecolor="#1f6f2a")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of frames")
    ax.set_title(title)
    ax.grid(True, color="0.88", linewidth=0.7)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_matched_frame_csv(output_path: Path, frames: list[int]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame"])
        for frame in frames:
            writer.writerow([frame])


def main() -> int:
    args = parse_args()
    if args.dry_run:
        model_dir, summary_glob, force_glob = expected_model_globs(args)
        status(f"Model: {args.model}")
        status(f"Expected MLIP directory: {model_dir}")
        status(f"Expected summary glob: {summary_glob}")
        status(f"Expected forces glob: {force_glob}")
        status(f"DFT outputs: {args.dft_dir}")
        status(f"Output directory: {args.output_dir / args.model}")
        if args.no_reference:
            status("Energy mode: raw total energies")
        else:
            status(f"Energy mode: atom-reference-subtracted using {args.atomic_reference}")
        return 0

    summary_csv, force_csv = model_paths(args)
    references = None if args.no_reference else load_atomic_references(args.atomic_reference)

    status(f"Model: {args.model}")
    status(f"MLIP summary: {summary_csv}")
    status(f"MLIP forces: {force_csv}")
    status(f"DFT outputs: {args.dft_dir}")
    if references is None:
        status("Energy mode: raw total energies")
    else:
        status(f"Energy mode: atom-reference-subtracted using {args.atomic_reference}")
    dft = load_dft_outputs(args.dft_dir)
    mlip_summary = load_mlip_summary(summary_csv)
    mlip_forces = load_mlip_forces(force_csv)
    force_abs, force_frac, energy_abs, energy_frac, frames = collect_errors(
        dft, mlip_summary, mlip_forces, references, args.fractional_eps, args.target_position
    )

    model_output_dir = args.output_dir / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    reference_label = "referenced" if references is not None else "raw total"
    prefix = f"{args.model}_"

    plot_force_histogram(
        force_abs,
        "Force error magnitude |F_MLIP - F_DFT| (eV/A)",
        f"{args.model}: absolute force errors ({len(frames)} frames)",
        model_output_dir / f"{prefix}absolute_force_error_hist.png",
        args.force_bins,
    )
    plot_force_histogram(
        force_frac,
        "Fractional force error |F_MLIP - F_DFT| / |F_DFT|",
        f"{args.model}: fractional force errors ({len(frames)} frames)",
        model_output_dir / f"{prefix}fractional_force_error_hist.png",
        args.force_bins,
    )
    plot_energy_histogram(
        energy_abs,
        f"Absolute {reference_label} energy error |E_MLIP - E_DFT| (eV)",
        f"{args.model}: absolute energy errors ({len(frames)} frames)",
        model_output_dir / f"{prefix}absolute_energy_error_hist.png",
        args.energy_bins,
    )
    plot_energy_histogram(
        energy_frac,
        f"Fractional {reference_label} energy error |E_MLIP - E_DFT| / |E_DFT|",
        f"{args.model}: fractional energy errors ({len(frames)} frames)",
        model_output_dir / f"{prefix}fractional_energy_error_hist.png",
        args.energy_bins,
    )
    write_matched_frame_csv(model_output_dir / f"{prefix}matched_frames.csv", frames)

    status(f"Matched {len(frames)} frame(s).")
    status(f"Wrote plots to {model_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
