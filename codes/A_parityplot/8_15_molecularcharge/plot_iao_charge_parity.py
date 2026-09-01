#!/usr/bin/env python3
"""Frame-by-frame charge evolution plots for DFT IAO and MLIP predicted charges."""

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
DEFAULT_DFT_DIR = SCRIPT_DIR.parents[2] / "outputsfull" / "8_5_bluehiveDFT"
DEFAULT_MLIP_ROOT = SCRIPT_DIR.parent / "8_6b_mlippredout2"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
EXPECTED_FRAMES = 180
DEFAULT_FRAMES_PER_INSTANCE = 10
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
            "values, and make frame-by-frame DFT and MLIP charge evolution plots."
        )
    )
    parser.add_argument("--dft-dir", type=Path, default=DEFAULT_DFT_DIR)
    parser.add_argument("--mlip-root", type=Path, default=DEFAULT_MLIP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES)
    parser.add_argument(
        "--frames-per-instance",
        type=int,
        default=DEFAULT_FRAMES_PER_INSTANCE,
        help="Number of frames in each independent H2-formation instance. Default: 10.",
    )
    parser.add_argument(
        "--focus-instance",
        type=int,
        default=None,
        help=(
            "Instance index to plot. Default: choose the instance with the clearest sustained H2 formation "
            "from the molecule assignments."
        ),
    )
    parser.add_argument(
        "--model",
        default="all",
        help="MLIP subdirectory to plot, e.g. polar1s, polar1m, off. Default: all.",
    )
    parser.add_argument("--dpi", type=int, default=400)
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


def collect_molecules(
    rows: list[dict[str, object]],
    bond_cutoff_a: float,
    frames_per_instance: int,
) -> list[dict[str, object]]:
    molecule_rows: list[dict[str, object]] = []
    by_frame: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame"]), []).append(row)

    for frame in sorted(by_frame):
        frame_rows = sorted(by_frame[frame], key=lambda row: int(row["atom_index"]))
        for molecule_index, component in enumerate(assign_molecules(frame_rows, bond_cutoff_a)):
            atom_indices = [int(row["atom_index"]) for row in component]
            symbols = [str(row["symbol"]) for row in component]
            species = molecule_species(symbols)
            oxygen_atom_indices = sorted(int(row["atom_index"]) for row in component if row["symbol"] == "O")
            hydrogen_atom_indices = sorted(int(row["atom_index"]) for row in component if row["symbol"] == "H")
            dft_charge = sum(float(row["dft_iao_charge_e"]) for row in component)
            mlip_charge = sum(float(row["mlip_predicted_charge_e"]) for row in component)
            raw_iao_charge = sum(float(row["dft_raw_iao_charge_e"]) for row in component)
            formal_charge = sum(float(row["formal_charge_e"]) for row in component)
            instance_index = frame // frames_per_instance
            local_frame = frame % frames_per_instance
            if species in {"HO", "H2O", "H3O"} and oxygen_atom_indices:
                molecule_track_id = f"O:{oxygen_atom_indices[0]}"
            elif species == "H2" and len(hydrogen_atom_indices) == 2:
                molecule_track_id = f"H2:{hydrogen_atom_indices[0]}-{hydrogen_atom_indices[1]}"
            elif species == "H" and len(hydrogen_atom_indices) == 1:
                molecule_track_id = f"H:{hydrogen_atom_indices[0]}"
            else:
                molecule_track_id = "atoms:" + "-".join(str(index) for index in atom_indices)
            molecule_rows.append(
                {
                    "model": str(component[0]["model"]),
                    "frame": frame,
                    "instance_index": instance_index,
                    "local_frame": local_frame,
                    "molecule_index": molecule_index,
                    "species": species,
                    "n_atoms": len(component),
                    "atom_indices": " ".join(str(index) for index in atom_indices),
                    "oxygen_atom_index": "" if not oxygen_atom_indices else oxygen_atom_indices[0],
                    "hydrogen_atom_indices": " ".join(str(index) for index in hydrogen_atom_indices),
                    "molecule_track_id": molecule_track_id,
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


def symbol_subsets(rows: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    symbols = sorted({str(row["symbol"]) for row in rows})
    return [(symbol, [row for row in rows if row["symbol"] == symbol]) for symbol in symbols]


def species_subsets(rows: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    species = sorted({str(row["species"]) for row in rows})
    return [(name, [row for row in rows if row["species"] == name]) for name in species]


def choose_focus_instance(rows: list[dict[str, object]]) -> int:
    by_instance: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_instance.setdefault(int(row["instance_index"]), []).append(row)

    candidates: list[tuple[int, int, int, int]] = []
    for instance_index, instance_rows in by_instance.items():
        h2_rows = [row for row in instance_rows if row["species"] == "H2"]
        if not h2_rows:
            continue
        local_frames = sorted({int(row["local_frame"]) for row in h2_rows})
        span = local_frames[-1] - local_frames[0]
        candidates.append((len(local_frames), span, -instance_index, instance_index))

    if not candidates:
        raise RuntimeError("No instance contains an H2 molecule; cannot choose a focused H2-formation run")
    return max(candidates)[3]


def charge_limits(rows: list[dict[str, object]], value_keys: list[str]) -> tuple[float, float]:
    values = [float(row[key]) for row in rows for key in value_keys]
    lo = min(values)
    hi = max(values)
    pad = 0.05 * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def plot_combined_molecule_charge_evolution(
    path: Path,
    dft_rows: list[dict[str, object]],
    mlip_rows_by_model: dict[str, list[dict[str, object]]],
    title: str,
    dft_value_key: str,
    mlip_value_key: str,
    ylabel: str,
    colors: dict[str, str],
    dpi: int,
    frames_per_instance: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 11.0), sharex=True, constrained_layout=True)
    all_rows = list(dft_rows)
    for rows in mlip_rows_by_model.values():
        all_rows.extend(rows)
    ylo, yhi = charge_limits(all_rows, [dft_value_key, mlip_value_key])
    panels = [
        (axes[0], "DFT", dft_rows, dft_value_key),
        (axes[1], "polar1m", mlip_rows_by_model["polar1m"], mlip_value_key),
        (axes[2], "polar1s", mlip_rows_by_model["polar1s"], mlip_value_key),
    ]
    for ax, panel_title, rows, value_key in panels:
        subsets: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            subsets.setdefault(str(row["species"]), []).append(row)
        used_labels: set[str] = set()
        for species in sorted(subsets):
            subset = sorted(
                subsets[species],
                key=lambda row: (
                    int(row["local_frame"]),
                    int(row["molecule_index"]),
                    int(row["frame"]),
                ),
            )
            label = species if species not in used_labels else None
            used_labels.add(species)
            ax.scatter(
                [int(row["local_frame"]) for row in subset],
                [float(row[value_key]) for row in subset],
                s=28,
                alpha=0.9,
                linewidth=0.2,
                edgecolor="white",
                color=colors.get(species, "#4c4c4c"),
                label=label,
            )
            traces: dict[str, list[dict[str, object]]] = {}
            for row in subset:
                traces.setdefault(str(row["molecule_track_id"]), []).append(row)
            for trace_rows in traces.values():
                ordered = sorted(trace_rows, key=lambda row: int(row["local_frame"]))
                if len(ordered) < 2:
                    continue
                ax.plot(
                    [int(row["local_frame"]) for row in ordered],
                    [float(row[value_key]) for row in ordered],
                    color=colors.get(species, "#4c4c4c"),
                    alpha=0.75,
                    linewidth=1.6,
                )
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylo, yhi)
        ax.set_title(panel_title)
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.legend(frameon=False, loc="best", ncol=2)

    axes[2].set_xlabel(f"Frame within 10-frame H2-formation instance (0-{frames_per_instance - 1})")
    axes[2].set_xlim(-0.25, frames_per_instance - 1 + 0.25)
    axes[2].set_xticks(list(range(frames_per_instance)))
    fig.suptitle(title)
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
    if args.frames_per_instance <= 0:
        raise RuntimeError("--frames-per-instance must be positive")
    if args.expected_frames % args.frames_per_instance != 0:
        raise RuntimeError(
            f"expected_frames={args.expected_frames} is not divisible by "
            f"frames_per_instance={args.frames_per_instance}"
        )

    status(f"Reading DFT IAO partial charges from {args.dft_dir}")
    dft, iao_sums = load_dft_iao_charges(args.dft_dir, args.expected_frames)
    write_iao_sum_csv(args.output_dir / "dft_iao_charge_sums.csv", iao_sums)
    status(f"Parsed {len(dft)} DFT atom charges across {len(iao_sums)} frames")

    focused_molecule_rows_by_model: dict[str, list[dict[str, object]]] = {}
    chosen_focus_instance: int | None = args.focus_instance
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
        molecule_rows = collect_molecules(rows, args.bond_cutoff, args.frames_per_instance)
        found_species = {str(row["species"]) for row in molecule_rows}
        unexpected_species = sorted(found_species - EXPECTED_SPECIES)
        missing_species = sorted(EXPECTED_SPECIES - found_species)
        if unexpected_species:
            status(f"Warning: {model_key} found unexpected species: {', '.join(unexpected_species)}")
        if missing_species:
            status(f"Warning: {model_key} did not find expected species: {', '.join(missing_species)}")

        write_joined_csv(args.output_dir / f"{model_key}_dft_iao_vs_mlip_charges.csv", rows)
        write_joined_csv(args.output_dir / f"{model_key}_molecule_dft_iao_vs_mlip_charges.csv", molecule_rows)
        if chosen_focus_instance is None:
            chosen_focus_instance = choose_focus_instance(molecule_rows)
        focused_molecule_rows = [row for row in molecule_rows if int(row["instance_index"]) == chosen_focus_instance]
        if not focused_molecule_rows:
            raise RuntimeError(f"{model_key} has no molecule rows for focus instance {chosen_focus_instance}")
        focus_frames = sorted(
            {int(row["local_frame"]) for row in focused_molecule_rows if str(row["species"]) == "H2"}
        )
        if focus_frames:
            status(
                f"Using {model_key} focus instance {chosen_focus_instance} with H2 on local frames "
                f"{focus_frames[0]}-{focus_frames[-1]}"
            )
        else:
            status(f"Using {model_key} focus instance {chosen_focus_instance}")
        focused_molecule_rows_by_model[model_key] = focused_molecule_rows

    if not focused_molecule_rows_by_model:
        raise RuntimeError("No MLIP models with numeric predicted_charge_e were plotted")
    required_models = {"polar1m", "polar1s"}
    missing_models = sorted(required_models - set(focused_molecule_rows_by_model))
    if missing_models:
        raise RuntimeError(f"Combined figure requires models {sorted(required_models)}; missing={missing_models}")
    if chosen_focus_instance is None:
        raise RuntimeError("Could not determine a focus instance")
    plot_combined_molecule_charge_evolution(
        args.output_dir / f"focused_instance_{chosen_focus_instance}_molecule_charge_evolution.png",
        focused_molecule_rows_by_model["polar1m"],
        {
            "polar1m": focused_molecule_rows_by_model["polar1m"],
            "polar1s": focused_molecule_rows_by_model["polar1s"],
        },
        f"Molecule charges for H2-formation instance {chosen_focus_instance}, cutoff={args.bond_cutoff:g} A",
        dft_value_key="dft_iao_charge_e",
        mlip_value_key="mlip_predicted_charge_e",
        ylabel="Molecular charge (e)",
        colors={
            "H": "#E69F00",
            "H2": "#56B4E9",
            "HO": "#8C564B",
            "H2O": "#9467BD",
            "H3O": "#CC79A7",
        },
        dpi=args.dpi,
        frames_per_instance=args.frames_per_instance,
    )
    status(f"Wrote plots and CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
