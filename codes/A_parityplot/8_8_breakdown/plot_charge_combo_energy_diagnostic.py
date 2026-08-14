#!/usr/bin/env python3
"""Make target-H breakdown plots colored by total system charge."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np


DFT_CHARGE_RE = re.compile(r"Total Charge\s+Charge\s+\.+\s+(-?\d+)")
CHARGE_COLORS = {
    -2: "#8c564b",
    -1: "#9467bd",
    0: "#1f77b4",
    1: "#2ca02c",
    2: "#ff7f0e",
    3: "#d62728",
}
FALLBACK_COLORS = ["#7f7f7f", "#17becf", "#bcbd22", "#e377c2", "#aec7e8"]
SIGNED_LONGITUDINAL_KEYS = {
    "force_error_longitudinal_eV_A",
    "force_error_longitudinal_fractional",
}

SCRIPT_DIR = Path(__file__).resolve().parent
BREAKDOWN_SCRIPT = SCRIPT_DIR / "plot_mace_off_target_h_breakdown.py"


def status(message: str) -> None:
    print(message, flush=True)


def load_breakdown_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("target_h_breakdown", BREAKDOWN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load breakdown helpers from {BREAKDOWN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_vector3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("must have three comma-separated values, e.g. 12,12,12")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def parse_args(breakdown: ModuleType) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make the same target-H breakdown scatter plots as plot_mace_off_target_h_breakdown.py, "
            "but color points by total system charge instead of nearest atom type."
        )
    )
    parser.add_argument("--model", choices=("all", *breakdown.MODEL_CONFIGS), default="all")
    parser.add_argument("--dft-dir", type=Path, default=breakdown.DEFAULT_DFT_DIR)
    parser.add_argument("--mlip-root", type=Path, default=breakdown.DEFAULT_MLIP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=breakdown.DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--atomic-reference",
        type=Path,
        default=breakdown.DEFAULT_ATOMIC_REFERENCE,
        help="CSV with Atom and DFT Atomization energies columns. Use --no-reference for raw total energies.",
    )
    parser.add_argument("--no-reference", action="store_true", help="Compare raw total energies.")
    parser.add_argument("--fractional-eps", type=float, default=breakdown.EPS)
    parser.add_argument(
        "--target-position",
        type=parse_vector3,
        default=(12.0, 12.0, 12.0),
        help="Position of the centered isolated H atom in Angstrom. Default: 12,12,12.",
    )
    parser.add_argument(
        "--charge-source",
        choices=("mlip", "dft", "formal"),
        default="mlip",
        help=(
            "mlip uses mlip_charge_setting_e from the new summary CSV; "
            "dft parses ORCA Total Charge; formal uses formal_charge_sum_e."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def charge_color(charge: int, fallback_assignments: dict[int, str]) -> str:
    if charge in CHARGE_COLORS:
        return CHARGE_COLORS[charge]
    if charge not in fallback_assignments:
        fallback_assignments[charge] = FALLBACK_COLORS[len(fallback_assignments) % len(FALLBACK_COLORS)]
    return fallback_assignments[charge]


def load_mlip_charges(summary_csv: Path, source: str) -> dict[int, int]:
    charges: dict[int, int] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            frame = int(row["frame"])
            if source == "mlip":
                if "mlip_charge_setting_e" not in row or row["mlip_charge_setting_e"] == "":
                    raise KeyError(
                        f"{summary_csv} has no mlip_charge_setting_e column. "
                        "Regenerate MLIP predictions with the fixed predict_singlepoints.py or use --charge-source formal."
                    )
                charges[frame] = int(round(float(row["mlip_charge_setting_e"])))
            elif source == "formal":
                charges[frame] = int(round(float(row["formal_charge_sum_e"])))
            else:
                raise ValueError(f"Unsupported MLIP charge source: {source}")
    if not charges:
        raise RuntimeError(f"No charges loaded from {summary_csv}")
    return charges


def load_dft_charges(dft_dir: Path, breakdown: ModuleType) -> dict[int, int]:
    charges: dict[int, int] = {}
    for path in sorted(dft_dir.glob("*.out")):
        frame = breakdown.frame_from_dft_path(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = DFT_CHARGE_RE.search(text)
        if match:
            charges[frame] = int(match.group(1))
    if not charges:
        raise RuntimeError(f"No DFT Total Charge lines found in {dft_dir}")
    return charges


def attach_charges(records: list[dict[str, object]], charges: dict[int, int]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for record in records:
        frame = int(record["frame"])
        if frame not in charges:
            continue
        enriched_record = dict(record)
        enriched_record["total_system_charge"] = int(charges[frame])
        enriched.append(enriched_record)
    if not enriched:
        raise RuntimeError("No records remained after matching total system charges")
    return enriched


def scatter_by_charge(ax: plt.Axes, records: list[dict[str, object]], y_key: str) -> None:
    fallback_assignments: dict[int, str] = {}
    charges = sorted({int(record["total_system_charge"]) for record in records})
    for charge in charges:
        subset = [record for record in records if int(record["total_system_charge"]) == charge]
        x = [float(record["nearest_separation_A"]) for record in subset]
        y = [float(record[y_key]) for record in subset]
        ax.scatter(
            x,
            y,
            s=36,
            alpha=0.8,
            color=charge_color(charge, fallback_assignments),
            edgecolor="white",
            linewidth=0.35,
            label=f"q={charge} (n={len(subset)})",
        )


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


def plot_charge_scatter(path: Path, records: list[dict[str, object]], y_key: str, ylabel: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.5), constrained_layout=True)
    scatter_by_charge(ax, records, y_key)
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


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def run_model(
    model_key: str,
    args: argparse.Namespace,
    breakdown: ModuleType,
    dft_outputs: dict[int, dict[str, object]] | None,
    references: dict[str, float] | None,
) -> None:
    summary_csv, force_csv = breakdown.model_paths(args.mlip_root, model_key)
    model_title = breakdown.MODEL_CONFIGS[model_key]["title"]
    output_dir = args.output_dir / model_key / "charge_colored"
    status(f"{model_title} summary: {summary_csv}")
    status(f"{model_title} forces: {force_csv}")
    status(f"{model_title} charge-colored output: {output_dir}")
    if args.dry_run:
        return
    if dft_outputs is None:
        raise RuntimeError("Internal error: DFT outputs were not loaded")

    mlip_summary = breakdown.load_mlip_summary(summary_csv)
    mlip_forces = breakdown.load_mlip_forces(force_csv)
    records = breakdown.collect_records(
        dft_outputs,
        mlip_summary,
        mlip_forces,
        references,
        args.target_position,
        args.fractional_eps,
    )

    if args.charge_source == "dft":
        charges = load_dft_charges(args.dft_dir, breakdown)
    else:
        charges = load_mlip_charges(summary_csv, args.charge_source)
    records = attach_charges(records, charges)

    csv_path = output_dir / f"{model_key}_target_h_breakdown_by_total_charge.csv"
    write_records(csv_path, records)
    for y_key, ylabel, filename in breakdown.PLOTS:
        plot_charge_scatter(
            output_dir / f"{model_key}_{filename}",
            records,
            y_key,
            ylabel,
            f"{model_title}: {ylabel} by total charge ({len(records)} frames)",
        )

    counts: dict[int, int] = {}
    for record in records:
        charge = int(record["total_system_charge"])
        counts[charge] = counts.get(charge, 0) + 1
    status(f"{model_title}: matched {len(records)} frame(s). Charge counts: {dict(sorted(counts.items()))}")
    status(f"{model_title}: wrote CSV and {len(breakdown.PLOTS)} plots to {output_dir}")


def main() -> int:
    breakdown = load_breakdown_module()
    args = parse_args(breakdown)
    model_keys = list(breakdown.MODEL_CONFIGS) if args.model == "all" else [args.model]
    references = None if args.no_reference else breakdown.load_atomic_references(args.atomic_reference)

    status(f"DFT outputs: {args.dft_dir}")
    status(f"MLIP root: {args.mlip_root}")
    status(f"Charge source: {args.charge_source}")
    status("Energy mode: raw total energies" if references is None else f"Energy mode: atom-reference-subtracted using {args.atomic_reference}")

    dft_outputs = None if args.dry_run else breakdown.load_dft_outputs(args.dft_dir)
    for model_key in model_keys:
        run_model(model_key, args, breakdown, dft_outputs, references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
