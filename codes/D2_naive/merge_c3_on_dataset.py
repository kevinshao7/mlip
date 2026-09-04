#!/usr/bin/env python3
"""Merge charge-audited C3 O/N calculations into D2's default dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_BASE = SCRIPT_DIR / "data" / "target_all.xyz"
DEFAULT_C3_ON = (
    MLIP_DIR
    / "outputsfull"
    / "C3_DFTproductionstopH2_ON"
    / "processed_dft_outputs"
    / "target_all.extxyz"
)
DEFAULT_OUTPUT = DEFAULT_BASE
FORMAL_CHARGES = {1: 1, 7: -3, 8: -2}
C3_DATASET_TAG = "C3_DFTproductionstopH2_ON"


def load_frames(path: Path) -> list[Atoms]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames = read(path, index=":")
    return [frames] if isinstance(frames, Atoms) else list(frames)


def formal_charge(frame: Atoms) -> int:
    unsupported = sorted(set(map(int, frame.numbers)) - FORMAL_CHARGES.keys())
    if unsupported:
        raise ValueError(f"Unsupported atomic numbers {unsupported}")
    return sum(FORMAL_CHARGES[int(number)] for number in frame.numbers)


def validate(frame: Atoms, label: str) -> None:
    recorded = int(frame.info["charge"])
    calculated = formal_charge(frame)
    if recorded != calculated:
        raise ValueError(f"{label}: recorded charge {recorded} != formal charge {calculated}")
    if "REF_energy" not in frame.info or not np.isfinite(float(frame.info["REF_energy"])):
        raise ValueError(f"{label}: missing or non-finite REF_energy")
    if "REF_forces" not in frame.arrays or frame.arrays["REF_forces"].shape != (len(frame), 3):
        raise ValueError(f"{label}: missing or malformed REF_forces")
    if not np.isfinite(frame.arrays["REF_forces"]).all():
        raise ValueError(f"{label}: non-finite REF_forces")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--c3-on", type=Path, default=DEFAULT_C3_ON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_frames(args.base)
    incoming = load_frames(args.c3_on)
    incoming_sources = {str(frame.info.get("source_file", "")) for frame in incoming}
    if "" in incoming_sources:
        raise ValueError("Every incoming frame must have source_file provenance")

    for index, frame in enumerate(incoming):
        validate(frame, f"incoming frame {index}")
        frame.info["source_dataset"] = C3_DATASET_TAG

    # Replace the same source calculations on rerun instead of accumulating copies.
    retained = [
        frame
        for frame in base
        if str(frame.info.get("source_file", "")) not in incoming_sources
        and str(frame.info.get("source_dataset", "")) != C3_DATASET_TAG
    ]
    merged = retained + incoming
    for index, frame in enumerate(merged):
        validate(frame, f"merged frame {index}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, merged, format="extxyz")
    with_n = sum(7 in frame.numbers for frame in incoming)
    print(
        f"base={len(base)} retained={len(retained)} incoming={len(incoming)} "
        f"incoming_with_N={with_n} merged={len(merged)} output={args.output}"
    )


if __name__ == "__main__":
    main()
