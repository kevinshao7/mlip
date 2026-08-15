#!/usr/bin/env python3
"""Parity plots for ORCA IAO partial charges vs MLIP predicted charges."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DFT_DIR = SCRIPT_DIR.parents[2] / "outputsfull" / "A_parityplot" / "8_5_bluehiveDFT"
DEFAULT_MLIP_ROOT = SCRIPT_DIR.parent / "8_6b_mlippredout2"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
EXPECTED_FRAMES = 180
DEFAULT_BOND_CUTOFF_A = 1.5
EXPECTED_SPECIES = {"H", "H2", "H2O", "HO", "H3O"}

FRAME_RE = re.compile(r"_(\d{3})\.out$")
IAO_LINE_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z][a-z]?)\s*:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*$")
IAO_SUM_RE = re.compile(r"Sum of atomic charges:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)")


@dataclass(frozen=True)
class DftCharge:
    frame: int
    atom_index: int
    symbol: str
    raw_iao_charge_e: float
    corrected_iao_charge_e: float


def status(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ORCA IAO PARTIAL CHARGES, join them to MLIP predicted_charge_e "
            "values, and make DFT-vs-MLIP parity plots."
        )
    )
    parser.add_argument("--dft-dir", type=Path, default=DEFAULT_DFT_DIR)
    parser.add_argument("--mlip-root", type=Path, default=DEFAULT_MLIP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    parser.add_argument(
        "--model",
        default="all",
        help="MLIP subdirectory to plot, e.g. polar1s, polar1m, off. Default: all.",
    )
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--bond-cutoff",
        type=float,
        default=DEFAULT_BOND_CUTOFF_A,
        help=(
            "Distance cutoff in Angstrom for molecule assignment. Each H attaches to its nearest O "
            "within this cutoff; unassigned H atoms are paired into H2 by mutual H-H distance."
        ),
    )
    return parser.parse_args()


def frame_from_out_path(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(match.group(1))


def parse_iao_charges(path: Path) -> tuple[list[DftCharge], float]:
    frame = frame_from_out_path(path)
    in_block = False
    saw_warning = False
    records: list[DftCharge] = []
    reported_sum: float | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not in_block:
                if line.strip() == "IAO PARTIAL CHARGES":
                    in_block = True
                continue

            if not saw_warning:
                if "Warning!!! IAOs HAVE MEANING" in line:
                    saw_warning = True
                continue

            sum_match = IAO_SUM_RE.search(line)
            if sum_match:
                reported_sum = float(sum_match.group(1))
                break

            charge_match = IAO_LINE_RE.match(line)
            if charge_match:
                symbol = charge_match.group(2)
                raw_charge = float(charge_match.group(3))
                corrected_charge = raw_charge - 2.0 if symbol == "O" else raw_charge
                records.append(
                    DftCharge(
                        frame=frame,
                        atom_index=int(charge_match.group(1)),
                        symbol=symbol,
                        raw_iao_charge_e=raw_charge,
                        corrected_iao_charge_e=corrected_charge,
                    )
                )

    if not records:
        raise RuntimeError(f"No IAO PARTIAL CHARGES block parsed from {path}")
    if reported_sum is None:
        raise RuntimeError(f"IAO PARTIAL CHARGES block in {path} had no sum line")
    return records, reported_sum


def load_dft_iao_charges(dft_dir: Path, expected_frames: int) -> tuple[dict[tuple[int, int], DftCharge], dict[int, float]]:
    paths = sorted(dft_dir.glob("*.out"))
    if len(paths) != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} DFT .out files in {dft_dir}, found {len(paths)}")

    dft: dict[tuple[int, int], DftCharge] = {}
    sums: dict[int, float] = {}
    for path in paths:
        records, reported_sum = parse_iao_charges(path)
        frame = records[0].frame
        sums[frame] = reported_sum
        for record in records:
            key = (record.frame, record.atom_index)
            if key in dft:
                raise RuntimeError(f"Duplicate DFT charge for frame {record.frame}, atom {record.atom_index}")
            dft[key] = record

    frames = sorted(sums)
    expected = list(range(expected_frames))
    if frames != expected:
        missing = sorted(set(expected) - set(frames))
        extra = sorted(set(frames) - set(expected))
        raise RuntimeError(f"DFT frame set is not 0..{expected_frames - 1}; missing={missing}, extra={extra}")
    return dft, sums


def discover_force_csvs(mlip_root: Path, model: str) -> list[Path]:
    if model == "all":
        paths = sorted(mlip_root.glob("*/*_forces.csv"))
    else:
        paths = sorted((mlip_root / model).glob("*_forces.csv"))
    if not paths:
        raise RuntimeError(f"No *_forces.csv files found for model={model!r} under {mlip_root}")
    return paths


def model_key_from_csv(path: Path) -> str:
    return path.parent.name


def join_mlip_to_dft(force_csv: Path, dft: dict[tuple[int, int], DftCharge]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    skipped_blank_predictions = 0
    with force_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "atom_index", "symbol", "x_A", "y_A", "z_A", "formal_charge_e", "predicted_charge_e"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{force_csv} is missing required columns: {sorted(missing)}")

        for row in reader:
            frame = int(row["frame"])
            atom_index = int(row["atom_index"])
            dft_record = dft.get((frame, atom_index))
            if dft_record is None:
                continue
            symbol = row["symbol"]
            if symbol != dft_record.symbol:
                raise RuntimeError(
                    f"Symbol mismatch in {force_csv}: frame {frame}, atom {atom_index}, "
                    f"MLIP={symbol}, DFT={dft_record.symbol}"
                )
            predicted_charge = row["predicted_charge_e"].strip()
            if predicted_charge == "":
                skipped_blank_predictions += 1
                continue
            mlip_charge = float(predicted_charge)
            dft_charge = dft_record.corrected_iao_charge_e
            rows.append(
                {
                    "model": model_key_from_csv(force_csv),
                    "frame": frame,
                    "atom_index": atom_index,
                    "symbol": symbol,
                    "x_A": float(row["x_A"]),
                    "y_A": float(row["y_A"]),
                    "z_A": float(row["z_A"]),
                    "formal_charge_e": float(row["formal_charge_e"]),
                    "dft_raw_iao_charge_e": dft_record.raw_iao_charge_e,
                    "dft_iao_charge_e": dft_charge,
                    "mlip_predicted_charge_e": mlip_charge,
                    "error_e": mlip_charge - dft_charge,
                    "abs_error_e": abs(mlip_charge - dft_charge),
                }
            )
    if not rows:
        raise RuntimeError(
            f"No joined DFT/MLIP rows with numeric predicted_charge_e for {force_csv} "
            f"(blank predictions skipped: {skipped_blank_predictions})"
        )
    return rows


def metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    dft = np.array([float(row["dft_iao_charge_e"]) for row in rows], dtype=float)
    mlip = np.array([float(row["mlip_predicted_charge_e"]) for row in rows], dtype=float)
    err = mlip - dft
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((dft - float(np.mean(dft))) ** 2))
    return {
        "n": float(len(rows)),
        "mae_e": float(np.mean(np.abs(err))),
        "rmse_e": float(math.sqrt(np.mean(err * err))),
        "bias_e": float(np.mean(err)),
        "r2": float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot,
    }


def molecule_species(symbols: list[str]) -> str:
    counts = {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}
    if set(counts) - {"H", "O"}:
        return "".join(f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}" for symbol in sorted(counts))
    h_count = counts.get("H", 0)
    o_count = counts.get("O", 0)
    if o_count == 0:
        return "H" if h_count == 1 else f"H{h_count}"
    if o_count == 1:
        h_part = "" if h_count == 0 else ("H" if h_count == 1 else f"H{h_count}")
        return f"{h_part}O"
    h_part = "" if h_count == 0 else ("H" if h_count == 1 else f"H{h_count}")
    return f"{h_part}O{o_count}"


def distance(row_a: dict[str, object], row_b: dict[str, object]) -> float:
    return math.sqrt(
        (float(row_a["x_A"]) - float(row_b["x_A"])) ** 2
        + (float(row_a["y_A"]) - float(row_b["y_A"])) ** 2
        + (float(row_a["z_A"]) - float(row_b["z_A"])) ** 2
    )


def assign_molecules(frame_rows: list[dict[str, object]], cutoff_a: float) -> list[list[dict[str, object]]]:
    oxygens = [row for row in frame_rows if row["symbol"] == "O"]
    hydrogens = [row for row in frame_rows if row["symbol"] == "H"]
    assigned_h: set[int] = set()
    by_oxygen: dict[int, list[dict[str, object]]] = {int(row["atom_index"]): [row] for row in oxygens}

    for hydrogen in hydrogens:
        candidates = [
            (distance(hydrogen, oxygen), oxygen)
            for oxygen in oxygens
            if distance(hydrogen, oxygen) <= cutoff_a
        ]
        if not candidates:
            continue
        _, oxygen = min(candidates, key=lambda item: (item[0], int(item[1]["atom_index"])))
        by_oxygen[int(oxygen["atom_index"])].append(hydrogen)
        assigned_h.add(int(hydrogen["atom_index"]))

    components = list(by_oxygen.values())
    unassigned_h = [row for row in hydrogens if int(row["atom_index"]) not in assigned_h]
    used_h: set[int] = set()
    cutoff_sq = cutoff_a * cutoff_a
    for hydrogen in unassigned_h:
        h_index = int(hydrogen["atom_index"])
        if h_index in used_h:
            continue
        partners = [
            other
            for other in unassigned_h
            if int(other["atom_index"]) != h_index
            and int(other["atom_index"]) not in used_h
            and distance(hydrogen, other) ** 2 <= cutoff_sq
        ]
        if partners:
            partner = min(partners, key=lambda other: (distance(hydrogen, other), int(other["atom_index"])))
            used_h.add(h_index)
            used_h.add(int(partner["atom_index"]))
            components.append([hydrogen, partner])
        else:
            used_h.add(h_index)
            components.append([hydrogen])

    return sorted(components, key=lambda component: min(int(row["atom_index"]) for row in component))


def collect_molecules(rows: list[dict[str, object]], bond_cutoff_a: float) -> list[dict[str, object]]:
    molecule_rows: list[dict[str, object]] = []
    by_frame: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame"]), []).append(row)

    for frame in sorted(by_frame):
        frame_rows = sorted(by_frame[frame], key=lambda row: int(row["atom_index"]))
        for molecule_index, component in enumerate(assign_molecules(frame_rows, bond_cutoff_a)):
            atom_indices = [int(row["atom_index"]) for row in component]
            symbols = [str(row["symbol"]) for row in component]
            dft_charge = sum(float(row["dft_iao_charge_e"]) for row in component)
            mlip_charge = sum(float(row["mlip_predicted_charge_e"]) for row in component)
            raw_iao_charge = sum(float(row["dft_raw_iao_charge_e"]) for row in component)
            formal_charge = sum(float(row["formal_charge_e"]) for row in component)
            molecule_rows.append(
                {
                    "model": str(component[0]["model"]),
                    "frame": frame,
                    "molecule_index": molecule_index,
                    "species": molecule_species(symbols),
                    "n_atoms": len(component),
                    "atom_indices": " ".join(str(index) for index in atom_indices),
                    "formula_symbols": " ".join(symbols),
                    "formal_charge_sum_e": formal_charge,
                    "dft_raw_iao_charge_sum_e": raw_iao_charge,
                    "dft_iao_charge_e": dft_charge,
                    "mlip_predicted_charge_e": mlip_charge,
                    "error_e": mlip_charge - dft_charge,
                    "abs_error_e": abs(mlip_charge - dft_charge),
                    "centroid_x_A": float(np.mean([float(row["x_A"]) for row in component])),
                    "centroid_y_A": float(np.mean([float(row["y_A"]) for row in component])),
                    "centroid_z_A": float(np.mean([float(row["z_A"]) for row in component])),
                }
            )
    return molecule_rows


def write_joined_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(path: Path, model_rows: dict[str, list[dict[str, object]]]) -> None:
    fieldnames = ["model", "subset", "n", "mae_e", "rmse_e", "bias_e", "r2"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_key, rows in sorted(model_rows.items()):
            for subset, subset_rows in [("all", rows), *symbol_subsets(rows)]:
                row = {"model": model_key, "subset": subset}
                row.update(metrics(subset_rows))
                writer.writerow(row)


def write_molecule_metrics_csv(path: Path, model_rows: dict[str, list[dict[str, object]]]) -> None:
    fieldnames = ["model", "subset", "n", "mae_e", "rmse_e", "bias_e", "r2"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_key, rows in sorted(model_rows.items()):
            subsets = [("all", rows), *species_subsets(rows)]
            for subset, subset_rows in subsets:
                row = {"model": model_key, "subset": subset}
                row.update(metrics(subset_rows))
                writer.writerow(row)


def symbol_subsets(rows: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    symbols = sorted({str(row["symbol"]) for row in rows})
    return [(symbol, [row for row in rows if row["symbol"] == symbol]) for symbol in symbols]


def species_subsets(rows: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    species = sorted({str(row["species"]) for row in rows})
    return [(name, [row for row in rows if row["species"] == name]) for name in species]


def axis_limits(rows: list[dict[str, object]]) -> tuple[float, float]:
    values = [
        *(float(row["dft_iao_charge_e"]) for row in rows),
        *(float(row["mlip_predicted_charge_e"]) for row in rows),
    ]
    lo = min(values)
    hi = max(values)
    pad = 0.05 * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def plot_parity(path: Path, rows: list[dict[str, object]], title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    colors = {"H": "#1f77b4", "O": "#d62728"}
    for symbol, subset in symbol_subsets(rows):
        ax.scatter(
            [float(row["dft_iao_charge_e"]) for row in subset],
            [float(row["mlip_predicted_charge_e"]) for row in subset],
            s=16,
            alpha=0.68,
            linewidth=0.2,
            edgecolor="white",
            color=colors.get(symbol, "#4c4c4c"),
            label=f"{symbol} (n={len(subset)})",
        )

    lo, hi = axis_limits(rows)
    ax.plot([lo, hi], [lo, hi], color="0.15", linestyle="--", linewidth=1.0, label="y=x")
    m = metrics(rows)
    ax.text(
        0.04,
        0.96,
        f"n={int(m['n'])}\nMAE={m['mae_e']:.4g} e\nRMSE={m['rmse_e']:.4g} e\nR2={m['r2']:.4g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.9, "pad": 4},
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("DFT IAO partial charge (e)")
    ax.set_ylabel("MLIP predicted charge (e)")
    ax.set_title(title)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_molecule_parity(path: Path, rows: list[dict[str, object]], title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.9), constrained_layout=True)
    colors = {
        "H": "#1f77b4",
        "H2": "#17becf",
        "HO": "#2ca02c",
        "H2O": "#d62728",
        "H3O": "#9467bd",
    }
    for species, subset in species_subsets(rows):
        ax.scatter(
            [float(row["dft_iao_charge_e"]) for row in subset],
            [float(row["mlip_predicted_charge_e"]) for row in subset],
            s=24,
            alpha=0.72,
            linewidth=0.25,
            edgecolor="white",
            color=colors.get(species, "#4c4c4c"),
            label=f"{species} (n={len(subset)})",
        )

    lo, hi = axis_limits(rows)
    ax.plot([lo, hi], [lo, hi], color="0.15", linestyle="--", linewidth=1.0, label="y=x")
    m = metrics(rows)
    ax.text(
        0.04,
        0.96,
        f"n={int(m['n'])}\nMAE={m['mae_e']:.4g} e\nRMSE={m['rmse_e']:.4g} e\nR2={m['r2']:.4g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.9, "pad": 4},
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("DFT molecule charge from corrected IAO sum (e)")
    ax.set_ylabel("MLIP molecule predicted charge sum (e)")
    ax.set_title(title)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_iao_sum_csv(path: Path, sums: dict[int, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "reported_iao_charge_sum_e"])
        writer.writeheader()
        for frame, value in sorted(sums.items()):
            writer.writerow({"frame": frame, "reported_iao_charge_sum_e": value})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    status(f"Reading DFT IAO partial charges from {args.dft_dir}")
    dft, iao_sums = load_dft_iao_charges(args.dft_dir, args.expected_frames)
    write_iao_sum_csv(args.output_dir / "dft_iao_charge_sums.csv", iao_sums)
    status(f"Parsed {len(dft)} DFT atom charges across {len(iao_sums)} frames")

    model_rows: dict[str, list[dict[str, object]]] = {}
    model_molecule_rows: dict[str, list[dict[str, object]]] = {}
    for force_csv in discover_force_csvs(args.mlip_root, args.model):
        model_key = model_key_from_csv(force_csv)
        status(f"Joining MLIP charges for {model_key}: {force_csv.name}")
        try:
            rows = join_mlip_to_dft(force_csv, dft)
        except RuntimeError as exc:
            if args.model == "all" and "numeric predicted_charge_e" in str(exc):
                status(f"Skipping {model_key}: {exc}")
                continue
            raise
        frames = sorted({int(row["frame"]) for row in rows})
        if frames != list(range(args.expected_frames)):
            missing = sorted(set(range(args.expected_frames)) - set(frames))
            raise RuntimeError(f"{model_key} joined frames are incomplete; missing={missing}")
        model_rows[model_key] = rows
        molecule_rows = collect_molecules(rows, args.bond_cutoff)
        found_species = {str(row["species"]) for row in molecule_rows}
        unexpected_species = sorted(found_species - EXPECTED_SPECIES)
        missing_species = sorted(EXPECTED_SPECIES - found_species)
        if unexpected_species:
            status(f"Warning: {model_key} found unexpected species: {', '.join(unexpected_species)}")
        if missing_species:
            status(f"Warning: {model_key} did not find expected species: {', '.join(missing_species)}")
        model_molecule_rows[model_key] = molecule_rows

        write_joined_csv(args.output_dir / f"{model_key}_dft_iao_vs_mlip_charges.csv", rows)
        write_joined_csv(args.output_dir / f"{model_key}_molecule_dft_iao_vs_mlip_charges.csv", molecule_rows)
        plot_molecule_parity(
            args.output_dir / f"{model_key}_molecule_dft_iao_vs_mlip_charge_parity.png",
            molecule_rows,
            f"{model_key} molecule charges, cutoff={args.bond_cutoff:g} A",
            args.dpi,
        )

    if not model_rows:
        raise RuntimeError("No MLIP models with numeric predicted_charge_e were plotted")
    write_metrics_csv(args.output_dir / "charge_parity_metrics.csv", model_rows)
    write_molecule_metrics_csv(args.output_dir / "molecule_charge_parity_metrics.csv", model_molecule_rows)
    status(f"Wrote plots and CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
