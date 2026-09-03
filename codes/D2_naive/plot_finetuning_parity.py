#!/usr/bin/env python3
"""Compare PolarMACE foundation and naïve-fine-tuned predictions with ORCA DFT.

The four PNG parity plots use one point per configuration for total system
energy and one point per Cartesian force component for every atom.  The most
erroneous observations are labelled in each plot and recorded in CSV files;
all configurations containing a labelled point are also written to extxyz for
structural inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# e3nn 0.4.x loads trusted package constants through torch.load; PyTorch 2.6
# otherwise rejects those bundled constants before either MACE model can load.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_MPL_CACHE = Path(__file__).resolve().parent / "runs" / ".matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import numpy as np
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
MACE_REPO = MLIP_DIR / "mace"
CACHE_DIR = MLIP_DIR / "outputsfull" / ".cache"
RUN_NAME = "polar1s_naive_orca_dft_e0"
RESULTS_DIR = SCRIPT_DIR / "runs" / RUN_NAME / "results"
DATA_DIR = MLIP_DIR / "codes" / "D_MHFT" / "data"
DEFAULT_SPLITS = {
    "train": DATA_DIR / "target_train.xyz",
    "valid": DATA_DIR / "target_valid.xyz",
    "test": DATA_DIR / "target_test.xyz",
}
DEFAULT_FINETUNED = SCRIPT_DIR / "runs" / RUN_NAME / "models" / f"{RUN_NAME}.model"
COMPONENT_NAMES = ("Fx", "Fy", "Fz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path,
                        help="Evaluate one DFT extxyz file instead of all shared train/valid/test splits.")
    parser.add_argument("--split-name", help=argparse.SUPPRESS)
    parser.add_argument("--finetuned-model", type=Path, default=DEFAULT_FINETUNED)
    parser.add_argument("--foundation-model", default="polar-1-s",
                        help="Polar model name, or a local foundation checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "parity")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--outliers", type=int, default=12,
                        help="Number of largest absolute-error points labelled per plot.")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Reuse existing predicted extxyz files in --output-dir.")
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path.resolve()


def resolve_foundation(model: str) -> Path:
    """Return a local PolarMACE checkpoint, downloading only when needed."""
    local_path = Path(model).expanduser()
    if local_path.is_file():
        return local_path.resolve()
    sys.path.insert(0, str(MACE_REPO.resolve()))
    from mace.calculators.foundations_models import download_mace_polar_checkpoint

    return Path(download_mace_polar_checkpoint(model)).resolve()


def evaluate(configs: Path, model: Path, output: Path, device: str) -> None:
    command = [
        sys.executable, str(SCRIPT_DIR / "evaluate.py"),
        "--configs", str(configs), "--model", str(model), "--output", str(output),
        "--device", device,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=os.environ.copy())


def config_label(frame, index: int) -> str:
    """Stable short label that links a point back to its ORCA source output."""
    source = Path(str(frame.info.get("source_file", "unknown"))).stem
    step = frame.info.get("orca_step", "?")
    return f"cfg{index}:{source}:step{step}"


def parity_plot(
    reference: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
    errors: np.ndarray,
    title: str,
    axis_label: str,
    output: Path,
    outlier_count: int,
) -> np.ndarray:
    """Write a PNG parity plot and return source-row indices of labelled points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = np.argsort(errors)[-min(outlier_count, len(errors)):][::-1]
    low, high = float(min(reference.min(), predicted.min())), float(max(reference.max(), predicted.max()))
    padding = max((high - low) * 0.04, 1.0e-8)
    low, high = low - padding, high + padding
    figure, axis = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    axis.scatter(reference, predicted, s=13, alpha=0.60, color="#0072B2", edgecolors="none")
    axis.plot((low, high), (low, high), color="#222222", linewidth=1.2, zorder=0)
    axis.scatter(reference[selected], predicted[selected], s=36, color="#D55E00", zorder=3)
    for row in selected:
        axis.annotate(labels[row], (reference[row], predicted[row]), xytext=(4, 4),
                      textcoords="offset points", fontsize=7, color="#7A3100")
    axis.set(xlim=(low, high), ylim=(low, high), xlabel=f"DFT {axis_label}",
             ylabel=f"Model {axis_label}", title=title)
    axis.set_aspect("equal", adjustable="box")
    rmse = float(np.sqrt(np.mean((predicted - reference) ** 2)))
    mae = float(np.mean(errors))
    axis.text(0.03, 0.97, f"N = {len(reference):,}\nRMSE = {rmse:.4g}\nMAE = {mae:.4g}",
              transform=axis.transAxes, va="top", fontsize=9,
              bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9})
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return selected


def main() -> None:
    args = parse_args()
    if args.outliers < 1:
        raise ValueError("--outliers must be positive")
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))
    if args.configs is None:
        # Keep each split self-contained: predictions, plots, tables, and
        # structures cannot be accidentally confused between train/valid/test.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        combined_summary: dict[str, object] = {}
        for split_name, split_configs in DEFAULT_SPLITS.items():
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--configs", str(split_configs), "--split-name", split_name,
                "--finetuned-model", str(args.finetuned_model),
                "--foundation-model", str(args.foundation_model),
                "--output-dir", str(args.output_dir / split_name),
                "--device", args.device, "--outliers", str(args.outliers),
            ]
            if args.skip_evaluation:
                command.append("--skip-evaluation")
            subprocess.run(command, check=True, env=os.environ.copy())
            summary_path = args.output_dir / split_name / "parity_summary.json"
            combined_summary[split_name] = json.loads(summary_path.read_text(encoding="utf-8"))
        (args.output_dir / "all_splits_summary.json").write_text(
            json.dumps(combined_summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote train, valid, and test parity analyses under {args.output_dir}")
        return
    configs = require_file(args.configs, "DFT configurations")
    finetuned = require_file(args.finetuned_model, "fine-tuned model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    foundation_prediction = args.output_dir / "foundation_predictions.xyz"
    finetuned_prediction = args.output_dir / "finetuned_predictions.xyz"
    if not args.skip_evaluation:
        foundation = resolve_foundation(args.foundation_model)
        evaluate(configs, foundation, foundation_prediction, args.device)
        evaluate(configs, finetuned, finetuned_prediction, args.device)
    require_file(foundation_prediction, "foundation prediction file")
    require_file(finetuned_prediction, "fine-tuned prediction file")

    dft_frames = list(read(configs, index=":"))
    foundation_frames = list(read(foundation_prediction, index=":"))
    finetuned_frames = list(read(finetuned_prediction, index=":"))
    if not (len(dft_frames) == len(foundation_frames) == len(finetuned_frames)):
        raise ValueError("Prediction files do not contain the same number of configurations as --configs")
    for index, (dft, foundation, tuned) in enumerate(zip(dft_frames, foundation_frames, finetuned_frames)):
        if not (len(dft) == len(foundation) == len(tuned)):
            raise ValueError(f"Atom-count mismatch in configuration {index}")

    dft_energy = np.array([frame.info["REF_energy"] for frame in dft_frames], dtype=float)
    foundation_energy = np.array([frame.info["MACE_energy"] for frame in foundation_frames], dtype=float)
    finetuned_energy = np.array([frame.info["MACE_energy"] for frame in finetuned_frames], dtype=float)
    dft_force = np.concatenate([frame.arrays["REF_forces"].reshape(-1) for frame in dft_frames])
    foundation_force = np.concatenate([frame.arrays["MACE_forces"].reshape(-1) for frame in foundation_frames])
    finetuned_force = np.concatenate([frame.arrays["MACE_forces"].reshape(-1) for frame in finetuned_frames])
    energy_labels = [config_label(frame, index) for index, frame in enumerate(dft_frames)]
    force_rows = [
        (config_index, atom_index, component)
        for config_index, frame in enumerate(dft_frames)
        for atom_index in range(len(frame)) for component in range(3)
    ]
    force_labels = [f"cfg{cfg}:atom{atom}:{COMPONENT_NAMES[component]}" for cfg, atom, component in force_rows]

    selected: dict[str, np.ndarray] = {}
    split_name = args.split_name or configs.stem
    selected["foundation_energy"] = parity_plot(
        dft_energy, foundation_energy, energy_labels, np.abs(foundation_energy - dft_energy),
        f"Foundation PolarMACE vs DFT ({split_name}): system energy", "system energy (eV)",
        args.output_dir / "foundation_energy_parity.png", args.outliers)
    selected["finetuned_energy"] = parity_plot(
        dft_energy, finetuned_energy, energy_labels, np.abs(finetuned_energy - dft_energy),
        f"Naïve-fine-tuned PolarMACE vs DFT ({split_name}): system energy", "system energy (eV)",
        args.output_dir / "finetuned_energy_parity.png", args.outliers)
    selected["foundation_force"] = parity_plot(
        dft_force, foundation_force, force_labels, np.abs(foundation_force - dft_force),
        f"Foundation PolarMACE vs DFT ({split_name}): force components", "force component (eV / Å)",
        args.output_dir / "foundation_force_parity.png", args.outliers)
    selected["finetuned_force"] = parity_plot(
        dft_force, finetuned_force, force_labels, np.abs(finetuned_force - dft_force),
        f"Naïve-fine-tuned PolarMACE vs DFT ({split_name}): force components", "force component (eV / Å)",
        args.output_dir / "finetuned_force_parity.png", args.outliers)

    outlier_configurations: set[int] = set()
    rows: list[dict[str, object]] = []
    for model_name, predicted_energy, predicted_force in (
        ("foundation", foundation_energy, foundation_force), ("finetuned", finetuned_energy, finetuned_force),
    ):
        for rank, index in enumerate(selected[f"{model_name}_energy"], start=1):
            outlier_configurations.add(int(index))
            rows.append({"model": model_name, "quantity": "system_energy", "rank": rank,
                         "configuration_index": int(index), "source_label": energy_labels[index],
                         "atom_index": "", "element": "", "component": "",
                         "dft_value": dft_energy[index], "model_value": predicted_energy[index],
                         "absolute_error": abs(predicted_energy[index] - dft_energy[index])})
        for rank, row_index in enumerate(selected[f"{model_name}_force"], start=1):
            cfg, atom, component = force_rows[row_index]
            outlier_configurations.add(cfg)
            rows.append({"model": model_name, "quantity": "force_component", "rank": rank,
                         "configuration_index": cfg, "source_label": energy_labels[cfg],
                         "atom_index": atom, "element": dft_frames[cfg].symbols[atom],
                         "component": COMPONENT_NAMES[component], "dft_value": dft_force[row_index],
                         "model_value": predicted_force[row_index],
                         "absolute_error": abs(predicted_force[row_index] - dft_force[row_index])})
    csv_path = args.output_dir / "labelled_outliers.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    structures = []
    for index in sorted(outlier_configurations):
        frame = dft_frames[index].copy()
        frame.info["parity_configuration_index"] = index
        frame.info["parity_source_label"] = energy_labels[index]
        structures.append(frame)
    write(args.output_dir / "labelled_outlier_structures.xyz", structures, format="extxyz")
    summary = {name: {"rmse": float(np.sqrt(np.mean((predicted - reference) ** 2))),
                      "mae": float(np.mean(np.abs(predicted - reference))), "n": len(reference)}
               for name, predicted, reference in (
                   ("foundation_system_energy_eV", foundation_energy, dft_energy),
                   ("finetuned_system_energy_eV", finetuned_energy, dft_energy),
                   ("foundation_force_component_eV_per_A", foundation_force, dft_force),
                   ("finetuned_force_component_eV_per_A", finetuned_force, dft_force),
               )}
    (args.output_dir / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote four PNG parity plots, {csv_path}, and {len(structures)} labelled outlier structures to {args.output_dir}")


if __name__ == "__main__":
    main()
