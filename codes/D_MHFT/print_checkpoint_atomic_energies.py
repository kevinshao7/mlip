#!/usr/bin/env python3
"""Print atomic energies embedded in a MACE foundation checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
MACE_REPO = MLIP_DIR / "mace"
DEFAULT_CHECKPOINT_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_polar_1/MACE-POLAR-1-S.model"
)


def download_model(checkpoint_url: str):
    if not MACE_REPO.is_dir():
        raise FileNotFoundError(f"Local MACE checkout is missing: {MACE_REPO}")

    sys.path.insert(0, str(MACE_REPO.resolve()))
    request = urllib.request.Request(
        checkpoint_url,
        headers={"User-Agent": "MACE-checkpoint-energy-reader"},
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint = Path(temporary_directory) / "checkpoint.model"
        with urllib.request.urlopen(request) as response, checkpoint.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        return torch.load(checkpoint, map_location="cpu", weights_only=False)


def extract_atomic_energies(model) -> dict[int, float]:
    if not hasattr(model, "atomic_energies_fn"):
        raise AttributeError("Checkpoint model has no atomic_energies_fn")
    if not hasattr(model, "atomic_numbers"):
        raise AttributeError("Checkpoint model has no atomic_numbers")

    atomic_numbers = model.atomic_numbers.detach().cpu().tolist()
    atomic_energies = model.atomic_energies_fn.atomic_energies.detach().cpu()
    if atomic_energies.ndim > 1:
        atomic_energies = atomic_energies.squeeze()
    if atomic_energies.ndim != 1:
        raise ValueError(
            f"Expected a 1D atomic energy array after squeeze, got shape {tuple(atomic_energies.shape)}"
        )
    if len(atomic_numbers) != len(atomic_energies):
        raise ValueError(
            f"Length mismatch: {len(atomic_numbers)} atomic numbers vs {len(atomic_energies)} energies"
        )

    return {
        int(z): float(energy)
        for z, energy in zip(atomic_numbers, atomic_energies.tolist(), strict=True)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-url",
        default=DEFAULT_CHECKPOINT_URL,
        help="Checkpoint URL to download afresh (default: MACE-POLAR-1-S release).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "lines"],
        default="json",
        help="Output format for the atomic energy table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = download_model(args.checkpoint_url)
    atomic_energies = extract_atomic_energies(model)

    if args.format == "json":
        print(json.dumps(atomic_energies, indent=2, sort_keys=True))
        return

    for atomic_number in sorted(atomic_energies):
        print(f"{atomic_number} {atomic_energies[atomic_number]:.15f}")


if __name__ == "__main__":
    main()
