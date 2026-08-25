#!/usr/bin/env python3
"""Plot molecular-species counts through time for production trajectories.

The default run analyses every condensed frame of all 20 manifest conditions,
processing conditions sequentially and frames with four worker processes. Raw
zero counts remain zero in the CSV; zeros are replaced by 0.1 only when drawing
the logarithmic plot. One CSV and one figure are written per condition.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One numerical thread per Python worker prevents nested oversubscription.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.formula import Formula

import plot_final_species_distribution as species_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "species_time_series"
# Production writes every 5 MD steps at 0.5 fs/step; condensed files retain
# every 100th written frame: 5 * 0.5 fs * 100 = 250 fs = 0.25 ps.
DEFAULT_FRAME_TIME_PS = 0.25
PLOT_ZERO_VALUE = 0.1
BASE_COLORS = ("#6f4e37", "#6b7280", "#111111")  # brown, grey, black
EXPECTED_COMPOSITIONS = (
    {"H": 4, "N": 1},  # NH4
    {"H": 3, "O": 1},  # H3O
    {"H": 1, "O": 1},  # HO/OH
    {"H": 2, "N": 1},  # NH2
)
BASE_COMPOSITIONS = (
    {"H": 2, "O": 1},  # H2O
    {"H": 3, "N": 1},  # NH3
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=species_base.DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=species_base.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-id",
        action="append",
        help="Restrict analysis to a condition; repeat for several. Default: all conditions.",
    )
    parser.add_argument("--all-conditions", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64, help="Frames submitted per parallel batch.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed frames (default: 10).",
    )
    parser.add_argument(
        "--frame-time-ps",
        type=float,
        default=DEFAULT_FRAME_TIME_PS,
        help="Time represented by one condensed-frame interval (default: 0.25 ps).",
    )
    parser.add_argument(
        "--min-peak-count",
        type=int,
        default=1,
        help="Plot species whose count reaches at least this value.",
    )
    parser.add_argument("--oh-cutoff", type=float, default=1.45)
    parser.add_argument("--nh-cutoff", type=float, default=1.30)
    parser.add_argument("--hh-cutoff", type=float, default=0.50)
    parser.add_argument("--bond-scale", type=float, default=1.20)
    return parser.parse_args()


def select_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = species_base.manifest_rows(args.manifest.resolve())
    if args.all_conditions:
        return rows
    if not args.run_id:
        return rows
    requested = set(args.run_id)
    selected = [row for row in rows if row["run_id"] in requested]
    missing = requested - {row["run_id"] for row in selected}
    if missing:
        raise ValueError(f"Run IDs absent from manifest: {', '.join(sorted(missing))}")
    return selected


def analyse_frame(
    task: tuple[int, species_base.Frame, float, float, float, float],
) -> tuple[int, dict[str, int]]:
    frame_index, frame, oh_cutoff, nh_cutoff, hh_cutoff, bond_scale = task
    components, _minimum_distance = species_base.molecular_components(
        frame, oh_cutoff, nh_cutoff, hh_cutoff, bond_scale
    )
    counts = Counter(species_base.component_formula(frame, component) for component in components)
    return frame_index, dict(counts)


def frame_tasks(
    trajectory: Path,
    args: argparse.Namespace,
):
    for frame_index, frame in species_base.iter_xyz_frames(trajectory):
        yield (
            frame_index,
            frame,
            args.oh_cutoff,
            args.nh_cutoff,
            args.hh_cutoff,
            args.bond_scale,
        )


def analyse_trajectory(
    trajectory: Path,
    args: argparse.Namespace,
    executor: ProcessPoolExecutor,
) -> list[tuple[int, dict[str, int]]]:
    """Stream frames from disk and process bounded batches in parallel."""
    results: list[tuple[int, dict[str, int]]] = []
    batch: list[tuple[int, species_base.Frame, float, float, float, float]] = []

    def process_batch(tasks: list[tuple[int, species_base.Frame, float, float, float, float]]) -> None:
        for result in executor.map(analyse_frame, tasks, chunksize=1):
            results.append(result)
            if len(results) == 1 or len(results) % args.progress_every == 0:
                print(
                    f"  Completed {len(results)} frame(s); "
                    f"latest condensed frame index={result[0]}",
                    flush=True,
                )

    for task in frame_tasks(trajectory, args):
        batch.append(task)
        if len(batch) == args.batch_size:
            process_batch(batch)
            batch.clear()
    if batch:
        process_batch(batch)
    if not results:
        raise ValueError(f"No frames found in {trajectory}")
    results.sort(key=lambda item: item[0])
    return results


def species_matrix(
    frame_results: list[tuple[int, dict[str, int]]],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    frame_indices = np.array([frame_index for frame_index, _counts in frame_results], dtype=int)
    names = sorted({name for _frame_index, counts in frame_results for name in counts})
    matrix = np.zeros((len(frame_results), len(names)), dtype=int)
    columns = {name: index for index, name in enumerate(names)}
    for row_index, (_frame_index, counts) in enumerate(frame_results):
        for name, count in counts.items():
            matrix[row_index, columns[name]] = count
    return frame_indices, names, matrix


def write_csv(
    path: Path,
    frame_indices: np.ndarray,
    times_ps: np.ndarray,
    names: list[str],
    counts: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condensed_frame_index_0based", "estimated_time_ps", *names])
        for frame_index, time_ps, row in zip(frame_indices, times_ps, counts, strict=True):
            writer.writerow([int(frame_index), f"{time_ps:.10g}", *map(int, row)])


def formula_counts(name: str) -> dict[str, int]:
    return dict(Formula(name).count())


def species_category(name: str) -> str:
    counts = formula_counts(name)
    if counts in BASE_COMPOSITIONS:
        return "base"
    if counts in EXPECTED_COMPOSITIONS:
        return "expected_ion"
    hydrogen = counts.get("H", 0)
    n_or_o = counts.get("N", 0) + counts.get("O", 0)
    if hydrogen > 0 and n_or_o > 0:
        return "unexpected_cluster"
    if (hydrogen > 0 and n_or_o == 0) or (hydrogen == 0 and n_or_o > 0):
        return "improper_species"
    return "improper_species"


def category_colors(names: list[str]) -> dict[str, tuple[float, float, float, float] | str]:
    """Assign deterministic colors from category-specific thirds of HSV."""
    grouped = {
        category: sorted(name for name in names if species_category(name) == category)
        for category in ("base", "expected_ion", "unexpected_cluster", "improper_species")
    }
    colors: dict[str, tuple[float, float, float, float] | str] = {}
    for index, name in enumerate(grouped["base"]):
        colors[name] = BASE_COLORS[index % len(BASE_COLORS)]
    wheel_sections = {
        "expected_ion": (0.00, 1.0 / 3.0),
        "unexpected_cluster": (1.0 / 3.0, 2.0 / 3.0),
        "improper_species": (2.0 / 3.0, 0.98),
    }
    color_map = matplotlib.colormaps["hsv"]
    for category, (start, stop) in wheel_sections.items():
        category_names = grouped[category]
        hues = np.linspace(start, stop, max(len(category_names), 2), endpoint=False)
        for name, hue in zip(category_names, hues, strict=False):
            colors[name] = color_map(float(hue))
    return colors


def plot_time_series(
    path: Path,
    row: dict[str, str],
    times_ps: np.ndarray,
    names: list[str],
    counts: np.ndarray,
    min_peak_count: int,
) -> None:
    selected = [index for index in range(len(names)) if counts[:, index].max() >= min_peak_count]
    if not selected:
        raise ValueError("No species satisfy --min-peak-count")
    category_order = {
        "base": 0,
        "expected_ion": 1,
        "unexpected_cluster": 2,
        "improper_species": 3,
    }
    selected.sort(key=lambda index: (category_order[species_category(names[index])], names[index]))

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    colors = category_colors([names[index] for index in selected])
    for index in selected:
        plot_values = np.where(counts[:, index] == 0, PLOT_ZERO_VALUE, counts[:, index])
        category = species_category(names[index]).replace("_", " ")
        ax.plot(
            times_ps,
            plot_values,
            linewidth=1.1,
            color=colors[names[index]],
            label=f"{names[index]} — {category}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Estimated production time (ps)")
    ax.set_ylabel("Species count (log scale; zero displayed at 0.1)")
    ax.set_title(
        f"Molecular-species evolution: {row['run_id']}\n"
        f"P={float(row['pressure_GPa']):g} GPa, T={float(row['temperature_K']):g} K, "
        f"NH3/H2O={float(row['ammonia_water_ratio']):g}"
    )
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    positive = (
        args.workers,
        args.batch_size,
        args.progress_every,
        args.frame_time_ps,
        args.min_peak_count,
        args.oh_cutoff,
        args.nh_cutoff,
        args.hh_cutoff,
        args.bond_scale,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Worker, batching, time, count, cutoff, and scale options must be positive")

    rows = select_rows(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Processing {len(rows)} condition(s) sequentially with "
        f"{args.workers} frame-analysis worker processes",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for condition_number, row in enumerate(rows, start=1):
            run_id = row["run_id"]
            trajectory = species_base.trajectory_for_run(args.input_dir.resolve(), run_id)
            print(
                f"[Condition {condition_number}/{len(rows)}: {run_id}] Reading {trajectory}",
                flush=True,
            )
            frame_results = analyse_trajectory(trajectory, args, executor)
            frame_indices, names, counts = species_matrix(frame_results)
            times_ps = frame_indices.astype(float) * args.frame_time_ps
            csv_path = output_dir / f"{run_id}_species_time_series.csv"
            plot_path = output_dir / f"{run_id}_species_time_series.png"
            write_csv(csv_path, frame_indices, times_ps, names, counts)
            plot_time_series(plot_path, row, times_ps, names, counts, args.min_peak_count)
            print(f"[{run_id}] Wrote {csv_path} and {plot_path}", flush=True)


if __name__ == "__main__":
    main()
