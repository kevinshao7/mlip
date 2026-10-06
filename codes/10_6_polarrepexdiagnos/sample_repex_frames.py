#!/usr/bin/env python3
"""Select a deterministic sample balanced across replica lambda and MD time."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write
from scipy.io import netcdf_file


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = MLIP_DIR / "outputsfull" / "9_29_repexfull"
DEFAULT_OUTPUT = MLIP_DIR / "outputsfull" / "10_6_polarrepexdiagnos" / "representative_100.xyz"
REPLICA_RE = re.compile(r"^replica_(\d+)_lambda_([0-9.]+)_el_([0-9.]+)$")


def fraction(value: str) -> float:
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise argparse.ArgumentTypeError("must satisfy 0 <= value < 1")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument(
        "--equilibration-fraction", type=fraction, default=0.20,
        help="Leading fraction omitted independently in every replica (default: 0.20).",
    )
    parser.add_argument(
        "--replicas", type=int, nargs="*",
        help="Optional replica indices; default uses every replica_* directory.",
    )
    return parser.parse_args()


def discover_replicas(input_dir: Path, requested: list[int] | None) -> list[tuple[int, float, float, Path]]:
    replicas = []
    requested_set = set(requested) if requested is not None else None
    for path in input_dir.glob("replica_*"):
        match = REPLICA_RE.match(path.name)
        if path.is_dir() and match:
            index = int(match.group(1))
            if requested_set is None or index in requested_set:
                replicas.append((index, float(match.group(2)), float(match.group(3)), path))
    replicas.sort(key=lambda item: item[0])
    if not replicas:
        raise FileNotFoundError(f"No selected replica directories found under {input_dir}")
    if requested_set is not None and {item[0] for item in replicas} != requested_set:
        found = {item[0] for item in replicas}
        raise FileNotFoundError(f"Requested replicas not found: {sorted(requested_set - found)}")
    return replicas


def allocate_counts(total: int, replica_count: int) -> list[int]:
    if total < replica_count:
        raise ValueError("--n-frames must be at least the number of selected replicas")
    base, remainder = divmod(total, replica_count)
    # Spread the extra frames across lambda space instead of assigning all to one end.
    extra = set(np.floor((np.arange(remainder) + 0.5) * replica_count / remainder).astype(int)) if remainder else set()
    return [base + int(index in extra) for index in range(replica_count)]


def evenly_spaced_indices(n_total: int, n_select: int, equilibration_fraction: float) -> np.ndarray:
    start = int(np.floor(n_total * equilibration_fraction))
    available = n_total - start
    if available < n_select:
        raise ValueError(f"Only {available} post-equilibration frames are available; requested {n_select}")
    # Midpoints of equal-width temporal strata avoid systematically selecting endpoints.
    offsets = np.floor((np.arange(n_select) + 0.5) * available / n_select).astype(int)
    return start + offsets


def read_selected_frames(
    replica: tuple[int, float, float, Path], indices: np.ndarray
) -> list[Atoms]:
    replica_index, lambda_value, lambda_el, replica_dir = replica
    template = read(replica_dir / "minimized.pdb", index=0)
    symbols = template.get_chemical_symbols()
    trajectory = replica_dir / "trajectory.nc"
    with netcdf_file(trajectory, "r", mmap=False) as dataset:
        coordinates = dataset.variables["coordinates"].data
        cell_lengths = dataset.variables["cell_lengths"].data
        cell_angles = dataset.variables["cell_angles"].data
        times = dataset.variables.get("time")
        frames = []
        for frame_index in indices:
            cellpar = np.concatenate((cell_lengths[frame_index], cell_angles[frame_index]))
            atoms = Atoms(
                symbols=symbols,
                positions=np.asarray(coordinates[frame_index], dtype=float),
                cell=cellpar_to_cell(cellpar),
                pbc=True,
            )
            atoms.info.update({
                "source_replica": replica_index,
                "source_lambda": lambda_value,
                "source_lambda_electrostatics": lambda_el,
                "source_frame": int(frame_index),
                "source_time_fs": float(times.data[frame_index]) if times is not None else float("nan"),
            })
            frames.append(atoms)
    return frames


def trajectory_length(path: Path) -> int:
    with netcdf_file(path, "r", mmap=False) as dataset:
        return int(dataset.variables["coordinates"].data.shape[0])


def main() -> None:
    args = parse_args()
    if args.n_frames < 1:
        raise ValueError("--n-frames must be positive")
    replicas = discover_replicas(args.input_dir.resolve(), args.replicas)
    counts = allocate_counts(args.n_frames, len(replicas))
    frames: list[Atoms] = []
    rows: list[dict[str, object]] = []
    for replica, count in zip(replicas, counts):
        n_total = trajectory_length(replica[3] / "trajectory.nc")
        indices = evenly_spaced_indices(n_total, count, args.equilibration_fraction)
        selected = read_selected_frames(replica, indices)
        frames.extend(selected)
        rows.extend(dict(frame.info) for frame in selected)

    if len(frames) != args.n_frames:
        raise RuntimeError(f"Internal sampling error: selected {len(frames)}, expected {args.n_frames}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, frames, format="extxyz")
    manifest = args.output.with_suffix(".csv")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(frames)} frames to {args.output}")
    print(f"Wrote sampling manifest to {manifest}")


if __name__ == "__main__":
    main()
