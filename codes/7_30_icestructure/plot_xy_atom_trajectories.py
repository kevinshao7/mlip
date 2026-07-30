#!/usr/bin/env python3
"""Plot xy trajectories for selected atoms from an extended XYZ trajectory."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Iterator
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputsfull" / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


DEFAULT_XYZ = (
    REPO_ROOT
    / "outputsfull"
    / "r09_hot_w"
    / "pressure_equil_seed_353168294_P_15GPa_T_300K_density_0.2.xyz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputsfull"
    / "r09_hot_w"
    / "ice_xy_atom_trajectories_4panel.png"
)
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


def parse_lattice(comment: str) -> np.ndarray:
    match = LATTICE_RE.search(comment)
    if not match:
        raise ValueError(f"Frame comment has no Lattice field: {comment[:120]}")
    values = np.fromstring(match.group(1), sep=" ", dtype=float)
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def read_first_frame_metadata(xyz_path: Path) -> tuple[list[str], np.ndarray]:
    with xyz_path.open("r", encoding="utf-8", errors="ignore") as handle:
        natoms_line = handle.readline()
        if not natoms_line:
            raise ValueError(f"{xyz_path} is empty")
        natoms = int(natoms_line.strip())
        comment = handle.readline()
        if not comment:
            raise ValueError(f"{xyz_path}: missing first-frame comment line")
        symbols = []
        for atom_index in range(natoms):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError(f"{xyz_path}: malformed first-frame atom line {atom_index}")
            symbols.append(fields[0])
    return symbols, parse_lattice(comment)


def default_atom_indices(symbols: list[str], element: str, count: int) -> list[int]:
    indices = [index for index, symbol in enumerate(symbols) if symbol == element]
    if len(indices) < count:
        raise ValueError(f"Found only {len(indices)} {element} atom(s), cannot plot {count}.")
    return indices[:count]


def iter_selected_xy(
    xyz_path: Path,
    atom_indices: list[int],
    stride: int,
    max_frames: int | None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    wanted = set(atom_indices)
    max_index = max(atom_indices)
    with xyz_path.open("r", encoding="utf-8", errors="ignore") as handle:
        frame_index = 0
        yielded = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue
            natoms = int(natoms_line)
            if max_index >= natoms:
                raise ValueError(f"Requested atom index {max_index}, but frame has {natoms} atoms.")
            comment = handle.readline()
            if not comment:
                raise ValueError(f"{xyz_path}: missing comment line at frame {frame_index}")

            xy_by_atom: dict[int, tuple[float, float]] = {}
            symbols_by_atom: dict[int, str] = {}
            for atom_index in range(natoms):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"{xyz_path}: malformed atom line at frame {frame_index}, atom {atom_index}")
                if atom_index in wanted:
                    symbols_by_atom[atom_index] = fields[0]
                    xy_by_atom[atom_index] = (float(fields[1]), float(fields[2]))

            if frame_index % stride == 0:
                xy = np.array([xy_by_atom[index] for index in atom_indices], dtype=float)
                symbols = np.array([symbols_by_atom[index] for index in atom_indices])
                yield frame_index, symbols, xy
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_index += 1


def unwrap_orthorhombic_xy(xy_tracks: np.ndarray, box_xy: np.ndarray) -> np.ndarray:
    unwrapped = xy_tracks.copy()
    for atom_col in range(unwrapped.shape[1]):
        deltas = np.diff(unwrapped[:, atom_col, :], axis=0)
        deltas -= np.round(deltas / box_xy) * box_xy
        unwrapped[1:, atom_col, :] = unwrapped[0, atom_col, :] + np.cumsum(deltas, axis=0)
    return unwrapped


def collect_tracks(
    xyz_path: Path,
    atom_indices: list[int],
    stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices: list[int] = []
    symbols: np.ndarray | None = None
    tracks: list[np.ndarray] = []
    for frame_index, frame_symbols, xy in iter_selected_xy(xyz_path, atom_indices, stride, max_frames):
        frame_indices.append(frame_index)
        symbols = frame_symbols
        tracks.append(xy)
    if not tracks:
        raise ValueError("No frames were read from the trajectory.")
    assert symbols is not None
    return np.array(frame_indices), symbols, np.stack(tracks)


def add_colored_path(ax: plt.Axes, xy: np.ndarray, values: np.ndarray, cmap: str) -> None:
    if len(xy) == 1:
        ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=22)
        return
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = 0.5 * (values[:-1] + values[1:])
    line = LineCollection(segments, array=colors, cmap=cmap, linewidth=1.1, alpha=0.95)
    ax.add_collection(line)
    ax.scatter(xy[0, 0], xy[0, 1], c=[values[0]], cmap=cmap, s=24, marker="o", edgecolor="black", linewidth=0.35)
    ax.scatter(xy[-1, 0], xy[-1, 1], c=[values[-1]], cmap=cmap, s=34, marker="*", edgecolor="black", linewidth=0.35)


def plot_tracks(
    frame_indices: np.ndarray,
    symbols: np.ndarray,
    xy_tracks: np.ndarray,
    atom_indices: list[int],
    cell: np.ndarray,
    output_path: Path,
    unwrap: bool,
    cmap: str,
) -> None:
    box_xy = np.array([cell[0, 0], cell[1, 1]], dtype=float)
    plot_tracks_xy = unwrap_orthorhombic_xy(xy_tracks, box_xy) if unwrap else xy_tracks

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for panel_index, ax in enumerate(axes_flat):
        xy = plot_tracks_xy[:, panel_index, :]
        add_colored_path(ax, xy, frame_indices, cmap)
        ax.set_title(f"{symbols[panel_index]} atom {atom_indices[panel_index]} ({atom_indices[panel_index] + 1} in XYZ)")
        ax.set_xlabel("x (Angstrom)")
        ax.set_ylabel("y (Angstrom)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.88", linewidth=0.6)
        if unwrap:
            pad = 0.75
            ax.set_xlim(float(np.min(xy[:, 0]) - pad), float(np.max(xy[:, 0]) + pad))
            ax.set_ylim(float(np.min(xy[:, 1]) - pad), float(np.max(xy[:, 1]) + pad))
        else:
            ax.set_xlim(0.0, box_xy[0])
            ax.set_ylim(0.0, box_xy[1])

    norm = plt.Normalize(vmin=float(frame_indices[0]), vmax=float(frame_indices[-1]))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_map, ax=axes_flat.tolist(), shrink=0.86, pad=0.02)
    colorbar.set_label("Trajectory frame")
    title_suffix = "unwrapped xy" if unwrap else "wrapped xy in periodic cell"
    fig.suptitle(f"Selected atom xy trajectories colored by time ({title_suffix})", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot four selected atom xy trajectories from a periodic extended XYZ trajectory."
    )
    parser.add_argument("--xyz", type=Path, default=DEFAULT_XYZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atom-indices", type=int, nargs=4, default=None, help="Four zero-based atom indices to plot.")
    parser.add_argument("--default-element", default="O", help="Element used for default atom selection.")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled frames to plot.")
    parser.add_argument("--unwrap", action="store_true", help="Unwrap xy jumps through an orthorhombic periodic cell.")
    parser.add_argument("--cmap", default="rainbow", help="Matplotlib colormap for time coloring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    symbols, cell = read_first_frame_metadata(args.xyz)
    atom_indices = args.atom_indices or default_atom_indices(symbols, args.default_element, 4)
    frame_indices, selected_symbols, xy_tracks = collect_tracks(args.xyz, atom_indices, args.stride, args.max_frames)
    plot_tracks(frame_indices, selected_symbols, xy_tracks, atom_indices, cell, args.output, args.unwrap, args.cmap)
    print(f"Read {len(frame_indices)} sampled frame(s) from {args.xyz}")
    print(f"Plotted zero-based atom indices: {', '.join(str(index) for index in atom_indices)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
