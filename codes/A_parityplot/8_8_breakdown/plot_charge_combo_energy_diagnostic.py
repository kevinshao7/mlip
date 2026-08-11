#!/usr/bin/env python3
"""Replot fractional per-atom energy error colored by MLIP/DFT charge pair."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DFT_FRAME_RE = re.compile(r"_(\d+)\.out$")
DFT_CHARGE_RE = re.compile(r"Total Charge\s+Charge\s+\.+\s+(-?\d+)")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DFT_DIR = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_5_bluehiveDFT"
DEFAULT_MLIP_ROOT = REPO_ROOT / "codes" / "A_parityplot" / "8_6b_mlippredout"
DEFAULT_BREAKDOWN_ROOT = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_8_breakdown"
MODEL_LABELS = {
    "polar1s": "mace_polar_1s",
    "polar1m": "mace_polar_1m",
    "off": "mace_off_medium",
}
COMBO_COLORS = {
    (0, 0): "#1f77b4",
    (1, 1): "#2ca02c",
    (0, 1): "#d62728",
    (1, 0): "#ff7f0e",
    (0, -1): "#9467bd",
    (0, 2): "#8c564b",
    (1, -1): "#e377c2",
    (1, 2): "#17becf",
}
FALLBACK_COLORS = ["#7f7f7f", "#bcbd22", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]


def status(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the separation-vs-signed-fractional-per-atom-energy-error plot, "
            "colored by MLIP system charge and DFT system charge."
        )
    )
    parser.add_argument("--model", choices=tuple(MODEL_LABELS), default="polar1m")
    parser.add_argument("--dft-dir", type=Path, default=DEFAULT_DFT_DIR)
    parser.add_argument("--mlip-root", type=Path, default=DEFAULT_MLIP_ROOT)
    parser.add_argument("--breakdown-root", type=Path, default=DEFAULT_BREAKDOWN_ROOT)
    parser.add_argument(
        "--mlip-charge-source",
        choices=("setting", "formal"),
        default="setting",
        help=(
            "setting uses --mlip-charge-setting for every frame, matching the MACE-POLAR run charge argument. "
            "formal uses formal_charge_sum_e from the MLIP summary CSV."
        ),
    )
    parser.add_argument(
        "--mlip-charge-setting",
        type=int,
        default=0,
        help="MLIP system charge setting used when --mlip-charge-source setting. Default: 0.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BREAKDOWN_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def frame_from_dft_path(path: Path) -> int:
    match = DFT_FRAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse frame from DFT output name: {path.name}")
    return int(match.group(1))


def load_dft_charges(dft_dir: Path) -> dict[int, int]:
    charges: dict[int, int] = {}
    for path in sorted(dft_dir.glob("*.out")):
        frame = frame_from_dft_path(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = DFT_CHARGE_RE.search(text)
        if match:
            charges[frame] = int(match.group(1))
            continue
        status(f"Skipping DFT charge for frame {frame}: no Total Charge line in {path.name}")
    if not charges:
        raise RuntimeError(f"No DFT charges found in {dft_dir}")
    return charges


def summary_path(mlip_root: Path, model: str) -> Path:
    label = MODEL_LABELS[model]
    matches = sorted((mlip_root / model).glob(f"*_{label}_singlepoints.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *_{label}_singlepoints.csv in {mlip_root / model}, found {len(matches)}")
    return matches[0]


def breakdown_path(breakdown_root: Path, model: str) -> Path:
    path = breakdown_root / model / f"{model}_target_h_breakdown.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Breakdown CSV not found: {path}")
    return path


def load_mlip_charges(args: argparse.Namespace) -> dict[int, int]:
    if args.mlip_charge_source == "setting":
        path = summary_path(args.mlip_root, args.model)
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {int(row["frame"]): int(args.mlip_charge_setting) for row in csv.DictReader(handle) if row.get("status") == "ok"}

    path = summary_path(args.mlip_root, args.model)
    charges: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            charges[int(row["frame"])] = int(round(float(row["formal_charge_sum_e"])))
    if not charges:
        raise RuntimeError(f"No MLIP formal charges loaded from {path}")
    return charges


def load_breakdown_records(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = {int(row["frame"]): row for row in csv.DictReader(handle)}
    if not records:
        raise RuntimeError(f"No breakdown rows found in {path}")
    return records


def combo_color(combo: tuple[int, int], assigned: dict[tuple[int, int], str]) -> str:
    if combo in COMBO_COLORS:
        return COMBO_COLORS[combo]
    if combo not in assigned:
        assigned[combo] = FALLBACK_COLORS[len(assigned) % len(FALLBACK_COLORS)]
    return assigned[combo]


def combo_label(combo: tuple[int, int]) -> str:
    mlip_charge, dft_charge = combo
    return f"MLIP q={mlip_charge}, DFT q={dft_charge}"


def collect_points(
    breakdown_records: dict[int, dict[str, str]],
    mlip_charges: dict[int, int],
    dft_charges: dict[int, int],
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for frame in sorted(set(breakdown_records) & set(mlip_charges) & set(dft_charges)):
        row = breakdown_records[frame]
        points.append(
            {
                "frame": frame,
                "nearest_separation_A": float(row["nearest_separation_A"]),
                "energy_error_per_atom_fractional": float(row["energy_error_per_atom_fractional"]),
                "nearest_symbol": row["nearest_symbol"],
                "mlip_charge": mlip_charges[frame],
                "dft_charge": dft_charges[frame],
            }
        )
    if not points:
        raise RuntimeError("No common frames between breakdown, MLIP charges, and DFT charges")
    return points


def write_points(path: Path, points: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0].keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(point)


def plot_charge_combo(path: Path, points: list[dict[str, object]], model: str, mlip_charge_source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 8.8), constrained_layout=True)
    fallback_assignments: dict[tuple[int, int], str] = {}
    combos = sorted({(int(p["mlip_charge"]), int(p["dft_charge"])) for p in points})

    for combo in combos:
        subset = [point for point in points if (int(point["mlip_charge"]), int(point["dft_charge"])) == combo]
        x = [float(point["nearest_separation_A"]) for point in subset]
        y = [float(point["energy_error_per_atom_fractional"]) for point in subset]
        ax.scatter(
            x,
            y,
            s=58,
            alpha=0.82,
            color=combo_color(combo, fallback_assignments),
            edgecolor="white",
            linewidth=0.45,
            label=f"{combo_label(combo)} (n={len(subset)})",
        )

    ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Separation between isolated H and nearest atom (A)")
    ax.set_ylabel("Signed fractional per-atom energy error")
    ax.set_title(f"{model}: charge-combo diagnostic ({len(points)} frames, MLIP charge source={mlip_charge_source})")
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = summary_path(args.mlip_root, args.model)
    breakdown = breakdown_path(args.breakdown_root, args.model)
    status(f"Model: {args.model}")
    status(f"MLIP summary: {summary}")
    status(f"Breakdown CSV: {breakdown}")
    status(f"DFT outputs: {args.dft_dir}")
    status(f"MLIP charge source: {args.mlip_charge_source}")
    if args.mlip_charge_source == "setting":
        status(f"MLIP charge setting: {args.mlip_charge_setting}")
    if args.dry_run:
        return 0

    dft_charges = load_dft_charges(args.dft_dir)
    mlip_charges = load_mlip_charges(args)
    breakdown_records = load_breakdown_records(breakdown)
    points = collect_points(breakdown_records, mlip_charges, dft_charges)

    output_base = args.output_dir / args.model
    source_suffix = f"mlip_charge_{args.mlip_charge_source}"
    csv_path = output_base / f"{args.model}_per_atom_energy_fractional_by_charge_combo_{source_suffix}.csv"
    plot_path = output_base / f"{args.model}_per_atom_energy_fractional_by_charge_combo_{source_suffix}.png"
    write_points(csv_path, points)
    plot_charge_combo(plot_path, points, args.model, args.mlip_charge_source)

    counts: dict[tuple[int, int], int] = {}
    for point in points:
        combo = (int(point["mlip_charge"]), int(point["dft_charge"]))
        counts[combo] = counts.get(combo, 0) + 1
    status(f"Matched {len(points)} frame(s). Charge combo counts: {dict(sorted(counts.items()))}")
    status(f"Wrote CSV: {csv_path}")
    status(f"Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
