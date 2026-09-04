#!/usr/bin/env python3
"""Create DFT-minus-Polar residual labels for a separate delta model.

The output labels are ``REF_energy = E_DFT - E_foundation`` and
``REF_forces = F_DFT - F_foundation``.  The original DFT and foundation labels
are retained as ``DFT_*`` and ``FOUNDATION_*`` fields, respectively.

By default this preserves the existing 32-structure training and 8-structure
validation splits.  ``--full-input`` additionally prepares residual labels
for a larger pool and can derive a training file by excluding a held-out
validation file.  A correction model trained on these files must be *added*
to the frozen foundation model at inference; it is not a drop-in target for
foundation-initialized fine tuning.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_FOUNDATION = Path("/home/kevinsh/mlip/outputsfull/.cache/mace/MACEPOLAR1Smodel")


def predict(input_path: Path, foundation_model: Path, prediction_path: Path, device: str) -> None:
    """Write MACE foundation predictions while preserving the DFT labels."""
    command = [
        sys.executable, str(SCRIPT_DIR / "evaluate.py"),
        "--configs", str(input_path.resolve()),
        "--model", str(foundation_model.resolve()),
        "--output", str(prediction_path.resolve()),
        "--device", device,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=os.environ.copy())


def make_residuals(input_path: Path, prediction_path: Path, output_path: Path) -> tuple[int, float, float]:
    """Subtract matching foundation predictions and write an extxyz residual set."""
    dft_frames = list(read(input_path, index=":"))
    predicted_frames = list(read(prediction_path, index=":"))
    if len(dft_frames) != len(predicted_frames):
        raise ValueError(f"Frame-count mismatch: {input_path} has {len(dft_frames)}, predictions have {len(predicted_frames)}")

    energy_residuals: list[float] = []
    force_residuals: list[np.ndarray] = []
    for index, (dft, prediction) in enumerate(zip(dft_frames, predicted_frames)):
        if len(dft) != len(prediction):
            raise ValueError(f"Atom-count mismatch in frame {index}")
        dft_energy = float(dft.info["REF_energy"])
        dft_forces = np.asarray(dft.arrays["REF_forces"], dtype=float)
        foundation_energy = float(prediction.info["MACE_energy"])
        foundation_forces = np.asarray(prediction.arrays["MACE_forces"], dtype=float)

        dft.info["DFT_energy"] = dft_energy
        dft.info["FOUNDATION_energy"] = foundation_energy
        dft.arrays["DFT_forces"] = dft_forces.copy()
        dft.arrays["FOUNDATION_forces"] = foundation_forces.copy()
        dft.info["REF_energy"] = dft_energy - foundation_energy
        dft.arrays["REF_forces"] = dft_forces - foundation_forces
        energy_residuals.append(float(dft.info["REF_energy"]))
        force_residuals.append(dft.arrays["REF_forces"].reshape(-1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, dft_frames, format="extxyz")
    return len(dft_frames), float(np.sqrt(np.mean(np.square(energy_residuals)))), float(
        np.sqrt(np.mean(np.square(np.concatenate(force_residuals))))
    )


def exclude_frames(full_delta_path: Path, excluded_input_path: Path, output_path: Path) -> int:
    """Write frames from ``full_delta_path`` not present in the held-out file.

    ``source_file`` is part of the dataset provenance and is a stable frame
    identifier here.  Checking uniqueness avoids accidentally dropping more
    than the intended validation structures.
    """
    frames = list(read(full_delta_path, index=":"))
    excluded = list(read(excluded_input_path, index=":"))
    excluded_ids = [str(frame.info["source_file"]) for frame in excluded]
    if len(set(excluded_ids)) != len(excluded_ids):
        raise ValueError("Held-out input contains duplicate source_file identifiers")
    frame_ids = [str(frame.info["source_file"]) for frame in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Full input contains duplicate source_file identifiers")
    missing = set(excluded_ids) - set(frame_ids)
    if missing:
        raise ValueError(f"Held-out structures are absent from full input: {sorted(missing)}")
    kept = [frame for frame, frame_id in zip(frames, frame_ids) if frame_id not in set(excluded_ids)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, kept, format="extxyz")
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=DATA_DIR / "target_train_32.xyz")
    parser.add_argument("--valid-input", type=Path, default=DATA_DIR / "target_valid.xyz")
    parser.add_argument("--train-output", type=Path, default=DATA_DIR / "target_train_32_delta.xyz")
    parser.add_argument("--valid-output", type=Path, default=DATA_DIR / "target_valid_delta.xyz")
    parser.add_argument("--foundation-model", type=Path, default=DEFAULT_FOUNDATION)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--full-input", type=Path,
                        help="Optional full DFT pool for a larger delta-learning experiment")
    parser.add_argument("--full-output", type=Path,
                        help="Residual output corresponding to --full-input")
    parser.add_argument("--exclude-input", type=Path,
                        help="Held-out structures to exclude when deriving --full-train-output")
    parser.add_argument("--full-train-output", type=Path,
                        help="Training residual output made from --full-output minus --exclude-input")
    args = parser.parse_args()
    if not args.foundation_model.is_file():
        raise FileNotFoundError(f"Missing frozen foundation model: {args.foundation_model}")

    for split, input_path, output_path in (
        ("train", args.train_input, args.train_output),
        ("valid", args.valid_input, args.valid_output),
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing {split} input: {input_path}")
        prediction_path = output_path.with_name(f"{output_path.stem}_foundation_predictions.xyz")
        predict(input_path, args.foundation_model, prediction_path, args.device)
        count, energy_rmse, force_rmse = make_residuals(input_path, prediction_path, output_path)
        print(f"Wrote {output_path}: {count} frames; foundation residual RMSE "
              f"E={energy_rmse * 1000:.3f} meV/config, F={force_rmse * 1000:.3f} meV/Å")

    full_arguments = (args.full_input, args.full_output, args.exclude_input, args.full_train_output)
    if any(value is not None for value in full_arguments):
        if not all(value is not None for value in full_arguments):
            parser.error("--full-input, --full-output, --exclude-input, and --full-train-output must be used together")
        if not args.full_input.is_file():
            raise FileNotFoundError(f"Missing full input: {args.full_input}")
        prediction_path = args.full_output.with_name(f"{args.full_output.stem}_foundation_predictions.xyz")
        predict(args.full_input, args.foundation_model, prediction_path, args.device)
        count, energy_rmse, force_rmse = make_residuals(args.full_input, prediction_path, args.full_output)
        print(f"Wrote {args.full_output}: {count} frames; foundation residual RMSE "
              f"E={energy_rmse * 1000:.3f} meV/config, F={force_rmse * 1000:.3f} meV/Å")
        kept = exclude_frames(args.full_output, args.exclude_input, args.full_train_output)
        print(f"Wrote {args.full_train_output}: {kept} frames after holding out "
              f"{count - kept} validation structures")


if __name__ == "__main__":
    main()
