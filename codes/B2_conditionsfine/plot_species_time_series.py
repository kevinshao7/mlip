#!/usr/bin/env python3
"""Run the production species-time-series analysis on B2 fine-grid outputs.

This is deliberately a thin adapter around
``B_conditionsproduction/plot_species_time_series.py``.  Reusing that module
keeps the molecular graph definition, species naming, CSV layout, plots, and
parallel processing exactly synchronized with the original analysis.
"""

from __future__ import annotations

import importlib
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


DISPLAY_SPECIES = ("H2O", "H3O", "HO", "NH3", "NH4", "NH2", "illegal")
COMPOSITION_LABELS = {
    (("H", 2), ("O", 1)): "H2O",
    (("H", 3), ("O", 1)): "H3O",
    (("H", 1), ("O", 1)): "HO",
    (("H", 3), ("N", 1)): "NH3",
    (("H", 4), ("N", 1)): "NH4",
    (("H", 2), ("N", 1)): "NH2",
}
SOURCE_TIME_RE = re.compile(r"(?:^|\s)source_time_fs=([0-9.eE+-]+)(?:\s|$)")
CURRENT_FRAME_TIMES_PS: dict[int, float] = {}
CURRENT_ORDERED_TIMES_PS = np.empty(0, dtype=float)


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
REFERENCE_DIR = SCRIPT_DIR.parent / "B_conditionsproduction"
REFERENCE_SCRIPT = REFERENCE_DIR / "plot_species_time_series.py"
DEFAULT_INPUT_DIR = MLIP_DIR / "outputsfull" / "B2a_conditionsfinereduced"
DEFAULT_MANIFEST = SCRIPT_DIR / "production_manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "species_time_series"
DEFAULT_SAMPLED_FRAMES_PER_CONDITION = 100


def load_reference_module():
    # The reference module imports plot_final_species_distribution by its
    # local module name, so its directory must precede other code directories.
    sys.path.insert(0, str(REFERENCE_DIR))
    # Use the real importable module name. ProcessPoolExecutor on Windows uses
    # spawn; a synthetic importlib name cannot be imported while unpickling
    # analyse_frame in the child and causes BrokenProcessPool.
    return importlib.import_module("plot_species_time_series")


def grouped_species_matrix(frame_results):
    """Collapse every non-requested molecular component into ``illegal``."""
    global CURRENT_ORDERED_TIMES_PS
    frame_indices = np.asarray([frame_index for frame_index, _counts in frame_results], dtype=int)
    try:
        CURRENT_ORDERED_TIMES_PS = np.asarray(
            [CURRENT_FRAME_TIMES_PS[int(index)] for index in frame_indices], dtype=float
        )
    except KeyError as exc:
        raise ValueError(f"No source_time_fs was recorded for reduced frame {exc.args[0]}") from exc
    matrix = np.zeros((len(frame_results), len(DISPLAY_SPECIES)), dtype=int)
    columns = {name: index for index, name in enumerate(DISPLAY_SPECIES)}
    for row_index, (_frame_index, raw_counts) in enumerate(frame_results):
        for formula, count in raw_counts.items():
            composition = tuple(sorted(Counter(dict_from_formula(formula)).items()))
            label = COMPOSITION_LABELS.get(composition, "illegal")
            matrix[row_index, columns[label]] += int(count)
    return frame_indices, list(DISPLAY_SPECIES), matrix


def dict_from_formula(formula: str) -> dict[str, int]:
    # ASE is already a required dependency of the reference analysis.
    from ase.formula import Formula

    return dict(Formula(formula).count())


def grouped_plot_time_series(path, row, times_ps, names, counts, _min_peak_count) -> None:
    """Plot exactly the six requested species plus their illegal aggregate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "H2O": "#4C78A8", "H3O": "#72B7B2", "HO": "#54A24B",
        "NH3": "#F58518", "NH4": "#ECA82C", "NH2": "#B279A2",
        "illegal": "#D62728",
    }
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    times_ps = CURRENT_ORDERED_TIMES_PS
    for index, name in enumerate(names):
        plot_values = np.where(counts[:, index] == 0, 0.1, counts[:, index])
        axis.plot(times_ps, plot_values, linewidth=1.2, color=colors[name], label=name)
    axis.set_yscale("log")
    axis.set_xlabel("Simulation time from trajectory metadata (ps)")
    axis.set_ylabel("Molecule count (log scale; zero displayed at 0.1)")
    axis.set_title(
        f"Selected molecular species: {row['run_id']}\n"
        f"P={float(row['pressure_GPa']):g} GPa, T={float(row['temperature_K']):g} K, "
        f"NH3/H2O={float(row['ammonia_water_ratio']):g}"
    )
    axis.grid(which="both", alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def grouped_write_csv(path, frame_indices, _times_ps, names, counts) -> None:
    """Write exact metadata times rather than an index-derived estimate."""
    if len(frame_indices) != len(CURRENT_ORDERED_TIMES_PS):
        raise ValueError("Frame/time length mismatch while writing species time series")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reduced_frame_index_0based", "simulation_time_ps", *names])
        for frame_index, time_ps, row in zip(
            frame_indices, CURRENT_ORDERED_TIMES_PS, counts, strict=True
        ):
            writer.writerow([int(frame_index), f"{float(time_ps):.12g}", *map(int, row)])


def sampled_frame_tasks(trajectory: Path, args):
    """Yield 100 approximately uniform frames while preserving source indices."""
    CURRENT_FRAME_TIMES_PS.clear()
    manifest = args.input_dir / "reduction_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Sampling requires the reduction manifest containing kept_frames: {manifest}"
        )
    run_id = trajectory.parent.name
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    if run_id not in rows:
        raise KeyError(f"Run {run_id} is absent from {manifest}")
    total_frames = int(rows[run_id]["kept_frames"])
    sample_count = min(DEFAULT_SAMPLED_FRAMES_PER_CONDITION, total_frames)
    selected = set(np.rint(np.linspace(0, total_frames - 1, sample_count)).astype(int).tolist())
    print(
        f"  Sampling {len(selected)}/{total_frames} frames uniformly across the trajectory",
        flush=True,
    )
    for frame_index, frame in args._species_base.iter_xyz_frames(trajectory):
        if frame_index in selected:
            match = SOURCE_TIME_RE.search(frame.comment)
            if match is None:
                raise ValueError(
                    f"Reduced frame {frame_index} in {trajectory} has no source_time_fs metadata"
                )
            CURRENT_FRAME_TIMES_PS[frame_index] = float(match.group(1)) / 1000.0
            yield (
                frame_index, frame, args.oh_cutoff, args.nh_cutoff,
                args.hh_cutoff, args.bond_scale,
            )
        if frame_index >= max(selected):
            break


def main() -> None:
    reference = load_reference_module()

    # parse_args reads these defaults when it is called.  No scientific or
    # plotting setting is changed: only B2 input/manifest/output locations.
    reference.species_base.DEFAULT_INPUT_DIR = DEFAULT_INPUT_DIR
    reference.species_base.DEFAULT_MANIFEST = DEFAULT_MANIFEST
    reference.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    reference.__doc__ = (
        "Plot molecular-species counts through time for all 15 B2 fine-grid "
        "conditions using the shared production molecule definition."
    )
    if "--workers" not in sys.argv and not any(arg.startswith("--workers=") for arg in sys.argv[1:]):
        sys.argv.extend(["--workers", "8"])
    reference.species_matrix = grouped_species_matrix
    reference.plot_time_series = grouped_plot_time_series
    reference.write_csv = grouped_write_csv
    # Pass the shared parser module without changing the inherited CLI object.
    original_parse_args = reference.parse_args

    def parse_args_with_species_base():
        args = original_parse_args()
        args._species_base = reference.species_base
        return args

    reference.parse_args = parse_args_with_species_base
    reference.frame_tasks = sampled_frame_tasks
    reference.main()


if __name__ == "__main__":
    main()
