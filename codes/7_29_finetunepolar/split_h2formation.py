#!/usr/bin/env python3
"""Create a reproducible train/validation/temporary-replay smoke-test split."""

from __future__ import annotations

import random
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "h2formation_orca.xyz"
DATA_DIR = ROOT / "data"
SEED = 42
SPLIT_SIZES = {
    "orca_train.xyz": 13,
    "orca_valid.xyz": 2,
    # Temporary ORCA stand-in until the actual PolarMACE replay set is available.
    "polar_replay.xyz": 2,
}


def main() -> None:
    frames = read(INPUT, index=":")
    expected = sum(SPLIT_SIZES.values())
    if len(frames) != expected:
        raise ValueError(
            f"Expected {expected} configurations in {INPUT}, found {len(frames)}."
        )

    indices = list(range(len(frames)))
    random.Random(SEED).shuffle(indices)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    offset = 0
    for filename, size in SPLIT_SIZES.items():
        selected = indices[offset : offset + size]
        output = DATA_DIR / filename
        write(output, [frames[index] for index in selected], format="extxyz")
        sources = [frames[index].info.get("source_file", index) for index in selected]
        print(f"{filename}: {size} configurations")
        for source in sources:
            print(f"  {source}")
        offset += size


if __name__ == "__main__":
    main()
