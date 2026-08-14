#!/usr/bin/env python3
"""Scatter MLIP errors against isolated-H nearest-neighbor separation."""

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
EPS = 1.0e-12

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DFT_DIR = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_5_bluehiveDFT"
DEFAULT_MLIP_ROOT = REPO_ROOT / "codes" / "A_parityplot" / "8_6b_mlippredout2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_8_breakdown"
DEFAULT_ATOMIC_REFERENCE = REPO_ROOT / "codes" / "7_7b_clustervalidation" / "atomizationenergies.txt"
MODEL_CONFIGS = {
    "off": {"label": "mace_off_medium", "title": "MACE-OFF"},
    "polar1s": {"label": "mace_polar_1s", "title": "MACE-POLAR 1s"},
    "polar1m": {"label": "mace_polar_1m", "title": "MACE-POLAR 1m"},
}
NEAREST_COLORS = {"O": "#1f77b4", "H": "#d62728"}
SIGNED_LONGITUDINAL_KEYS = {
    "force_error_longitudinal_eV_A",
    "force_error_longitudinal_fractional",
}


def status(message: str) -> None:
    print(message, flush=True)


def parse_vector3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("must have three comma-separated values, e.g. 12,12,12")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MLIP target-H force and energy errors versus separation "
            "to the nearest other atom."
        )
    )
    parser.add_argument("--model", choices=("all", *MODEL_CONFIGS), default="all")
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
    parser.add_argument("--fractional-eps", type=float, default=EPS)
    parser.add_argument(
        "--target-position",
        type=parse_vector3,
        default=(12.0, 12.0, 12.0),
        help="Position of the centered isolated H atom in Angstrom. Default: 12,12,12.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def model_paths(mlip_root: Path, model_key: str) -> tuple[Path, Path]:
    label = MODEL_CONFIGS[model_key]["label"]
    model_dir = mlip_root / model_key
    summary_matches = sorted(model_dir.glob(f"*_{label}_singlepoints.csv"))
    force_matches = sorted(model_dir.glob(f"*_{label}_forces.csv"))
    if len(summary_matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *_{label}_singlepoints.csv in {model_dir}, found {len(summary_matches)}")
    if len(force_matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *_{label}_forces.csv in {model_dir}, found {len(force_matches)}")
    return summary_matches[0], force_matches[0]


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

    gradient_blocks: list[list[tuple[str, list[float]]]] = []
    lines = text.splitlines()
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
    records: dict[int, dict[str, object]] = {}
    for path in sorted(dft_dir.glob("*.out")):
        frame = frame_from_dft_path(path)
        try:
            energy_ev, symbols, forces_ev_a = parse_dft_output(path)
        except Exception as exc:
            status(f"Skipping DFT frame {frame}: {exc}")
            continue
        records[frame] = {"energy_ev": energy_ev, "symbols": symbols, "forces": forces_ev_a}
    if not records:
        raise RuntimeError(f"No complete DFT outputs found in {dft_dir}")
    return records


def load_mlip_summary(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {int(row["frame"]): row for row in reader if row.get("status") == "ok"}
    if not rows:
        raise RuntimeError(f"No successful MLIP summary rows found in {path}")
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


def mlip_arrays(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    symbols = [row["symbol"] for row in rows]
    positions = np.array([[float(row["x_A"]), float(row["y_A"]), float(row["z_A"])] for row in rows], dtype=float)
    forces = np.array(
        [[float(row["force_x_eV_A"]), float(row["force_y_eV_A"]), float(row["force_z_eV_A"])] for row in rows],
        dtype=float,
    )
    return symbols, positions, forces


def target_and_nearest(
    symbols: list[str],
    positions: np.ndarray,
    target_position: tuple[float, float, float],
) -> tuple[int, int, float, np.ndarray]:
    target_point = np.asarray(target_position, dtype=float)
    hydrogen_indices = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    if not hydrogen_indices:
        raise ValueError("No H atoms found; cannot identify centered isolated H")

    target_distances = np.array([np.linalg.norm(positions[index] - target_point) for index in hydrogen_indices])
    target_index = hydrogen_indices[int(np.argmin(target_distances))]
    if float(np.min(target_distances)) > 1.0e-5:
        raise ValueError(
            f"No H atom found at {target_position}; closest H index {target_index} "
            f"is {float(np.min(target_distances)):.6g} A away"
        )

    separations = np.linalg.norm(positions - positions[target_index], axis=1)
    separations[target_index] = math.inf
    nearest_index = int(np.argmin(separations))
    separation = float(separations[nearest_index])
    if separation <= 0.0 or not np.isfinite(separation):
        raise ValueError("Could not compute isolated-H nearest-neighbor direction")
    unit_vector = (positions[nearest_index] - positions[target_index]) / separation
    return target_index, nearest_index, separation, unit_vector


def force_component_errors(
    mlip_force: np.ndarray,
    dft_force: np.ndarray,
    unit_vector: np.ndarray,
    fractional_eps: float,
) -> dict[str, float]:
    error = mlip_force - dft_force
    error_longitudinal = float(np.dot(error, unit_vector))
    dft_longitudinal = float(np.dot(dft_force, unit_vector))
    error_perp_vec = error - error_longitudinal * unit_vector
    dft_perp_vec = dft_force - dft_longitudinal * unit_vector
    abs_longitudinal = abs(error_longitudinal)
    abs_perpendicular = float(np.linalg.norm(error_perp_vec))
    return {
        "force_error_longitudinal_eV_A": error_longitudinal,
        "force_error_longitudinal_abs_eV_A": abs_longitudinal,
        "force_error_longitudinal_fractional": error_longitudinal / max(abs(dft_longitudinal), fractional_eps),
        "force_error_perpendicular_abs_eV_A": abs_perpendicular,
        "force_error_perpendicular_fractional": abs_perpendicular / max(float(np.linalg.norm(dft_perp_vec)), fractional_eps),
        "dft_force_longitudinal_eV_A": dft_longitudinal,
        "dft_force_perpendicular_eV_A": float(np.linalg.norm(dft_perp_vec)),
    }


def collect_records(
    dft_outputs: dict[int, dict[str, object]],
    mlip_summary: dict[int, dict[str, str]],
    mlip_forces: dict[int, list[dict[str, str]]],
    references: dict[str, float] | None,
    target_position: tuple[float, float, float],
    fractional_eps: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    common_frames = sorted(set(dft_outputs) & set(mlip_summary) & set(mlip_forces))
    for frame in common_frames:
        dft_record = dft_outputs[frame]
        dft_symbols = list(dft_record["symbols"])  # type: ignore[arg-type]
        dft_forces = np.asarray(dft_record["forces"], dtype=float)
        mlip_symbols, positions, mlip_force_array = mlip_arrays(mlip_forces[frame])
        if mlip_symbols != dft_symbols:
            raise ValueError(f"Atom order mismatch in frame {frame}")
        if mlip_force_array.shape != dft_forces.shape:
            raise ValueError(f"Force shape mismatch in frame {frame}: MLIP {mlip_force_array.shape}, DFT {dft_forces.shape}")

        target_index, nearest_index, separation, unit_vector = target_and_nearest(mlip_symbols, positions, target_position)
        force_errors = force_component_errors(
            mlip_force_array[target_index],
            dft_forces[target_index],
            unit_vector,
            fractional_eps,
        )

        dft_energy = float(dft_record["energy_ev"])
        mlip_energy = float(mlip_summary[frame]["energy_eV"])
        if references is not None:
            ref_sum = reference_sum(dft_symbols, references)
            dft_energy -= ref_sum
            mlip_energy -= ref_sum
        natoms = len(dft_symbols)
        total_energy_error = mlip_energy - dft_energy
        per_atom_energy_error = total_energy_error / natoms

        records.append(
            {
                "frame": frame,
                "natoms": natoms,
                "target_atom_index": target_index,
                "nearest_atom_index": nearest_index,
                "nearest_symbol": mlip_symbols[nearest_index],
                "nearest_separation_A": separation,
                "energy_error_total_eV": total_energy_error,
                "energy_error_total_fractional": total_energy_error / max(abs(dft_energy), fractional_eps),
                "energy_error_per_atom_eV_atom": per_atom_energy_error,
                "energy_error_per_atom_fractional": per_atom_energy_error / max(abs(dft_energy / natoms), fractional_eps),
                **force_errors,
            }
        )
    if not records:
        raise RuntimeError("No common complete frames found between MLIP and DFT")
    return records


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def scatter_by_nearest(ax: plt.Axes, records: list[dict[str, object]], y_key: str) -> None:
    for symbol, color in NEAREST_COLORS.items():
        subset = [record for record in records if record["nearest_symbol"] == symbol]
        if not subset:
            continue
        x = [float(record["nearest_separation_A"]) for record in subset]
        y = [float(record[y_key]) for record in subset]
        ax.scatter(x, y, s=34, alpha=0.78, color=color, edgecolor="white", linewidth=0.35, label=f"nearest {symbol}")

    other_symbols = sorted({str(record["nearest_symbol"]) for record in records} - set(NEAREST_COLORS))
    for symbol in other_symbols:
        subset = [record for record in records if record["nearest_symbol"] == symbol]
        x = [float(record["nearest_separation_A"]) for record in subset]
        y = [float(record[y_key]) for record in subset]
        ax.scatter(x, y, s=34, alpha=0.78, color="#7f7f7f", edgecolor="white", linewidth=0.35, label=f"nearest {symbol}")


def annotate_outlier_frames(ax: plt.Axes, records: list[dict[str, object]], y_key: str, count: int = 3) -> None:
    finite_records = [record for record in records if np.isfinite(float(record[y_key]))]
    outliers = sorted(finite_records, key=lambda record: abs(float(record[y_key])), reverse=True)[:count]
    for offset, record in enumerate(outliers):
        x = float(record["nearest_separation_A"])
        y = float(record[y_key])
        ax.annotate(
            f"f{int(record['frame'])}",
            xy=(x, y),
            xytext=(6, 7 + 8 * offset),
            textcoords="offset points",
            fontsize=8,
            color="0.12",
            arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.65},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        )


def plot_scatter(path: Path, records: list[dict[str, object]], y_key: str, ylabel: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.5), constrained_layout=True)
    scatter_by_nearest(ax, records, y_key)
    if y_key in SIGNED_LONGITUDINAL_KEYS:
        ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Separation between isolated H and nearest atom (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False)
    annotate_outlier_frames(ax, records, y_key)
    fig.savefig(path, dpi=220)
    plt.close(fig)


PLOTS = [
    (
        "force_error_longitudinal_eV_A",
        "Signed isolated-H longitudinal force error (eV/A)\npositive = MLIP more attractive than DFT",
        "target_h_longitudinal_force_error_abs.png",
    ),
    (
        "force_error_longitudinal_fractional",
        "Signed isolated-H fractional longitudinal force error\npositive = MLIP more attractive than DFT",
        "target_h_longitudinal_force_error_fractional.png",
    ),
    (
        "force_error_perpendicular_abs_eV_A",
        "Isolated-H perpendicular force error magnitude (eV/A)",
        "target_h_perpendicular_force_error_abs.png",
    ),
    (
        "force_error_perpendicular_fractional",
        "Isolated-H fractional perpendicular force error",
        "target_h_perpendicular_force_error_fractional.png",
    ),
    (
        "energy_error_total_eV",
        "Signed total energy error E_MLIP - E_DFT (eV)",
        "total_energy_error_signed.png",
    ),
    (
        "energy_error_total_fractional",
        "Signed fractional total energy error",
        "total_energy_error_fractional.png",
    ),
    (
        "energy_error_per_atom_eV_atom",
        "Signed per-atom energy error (eV/atom)",
        "per_atom_energy_error_signed.png",
    ),
    (
        "energy_error_per_atom_fractional",
        "Signed fractional per-atom energy error",
        "per_atom_energy_error_fractional.png",
    ),
]


def run_model(
    model_key: str,
    args: argparse.Namespace,
    dft_outputs: dict[int, dict[str, object]] | None,
    references: dict[str, float] | None,
) -> None:
    summary_csv, force_csv = model_paths(args.mlip_root, model_key)
    model_title = MODEL_CONFIGS[model_key]["title"]
    model_output_dir = args.output_dir / model_key
    status(f"{model_title} summary: {summary_csv}")
    status(f"{model_title} forces: {force_csv}")
    status(f"{model_title} output: {model_output_dir}")
    if args.dry_run:
        return
    if dft_outputs is None:
        raise RuntimeError("Internal error: DFT outputs were not loaded")

    mlip_summary = load_mlip_summary(summary_csv)
    mlip_forces = load_mlip_forces(force_csv)
    records = collect_records(
        dft_outputs,
        mlip_summary,
        mlip_forces,
        references,
        args.target_position,
        args.fractional_eps,
    )

    csv_path = model_output_dir / f"{model_key}_target_h_breakdown.csv"
    write_records(csv_path, records)
    for y_key, ylabel, filename in PLOTS:
        plot_scatter(
            model_output_dir / f"{model_key}_{filename}",
            records,
            y_key,
            ylabel,
            f"{model_title}: {ylabel} ({len(records)} frames)",
        )

    counts: dict[str, int] = {}
    for record in records:
        symbol = str(record["nearest_symbol"])
        counts[symbol] = counts.get(symbol, 0) + 1
    status(f"{model_title}: matched {len(records)} frame(s). Nearest atom counts: {counts}")
    status(f"{model_title}: wrote CSV and {len(PLOTS)} plots to {model_output_dir}")


def main() -> int:
    args = parse_args()
    model_keys = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    references = None if args.no_reference else load_atomic_references(args.atomic_reference)

    status(f"DFT outputs: {args.dft_dir}")
    status(f"MLIP root: {args.mlip_root}")
    status("Energy mode: raw total energies" if references is None else f"Energy mode: atom-reference-subtracted using {args.atomic_reference}")

    dft_outputs = None if args.dry_run else load_dft_outputs(args.dft_dir)
    for model_key in model_keys:
        run_model(model_key, args, dft_outputs, references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
