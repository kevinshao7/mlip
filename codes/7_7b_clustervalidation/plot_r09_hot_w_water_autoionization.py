from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "outputsfull" / "r09_hot_w"
FS_PER_PS = 1000.0
SPECIES = ("O", "OH", "H2O", "H3O", "H4O_or_more")
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


def find_one(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return files[0] if files else None


def load_thermo(thermo_path: Path) -> dict[str, np.ndarray]:
    if not thermo_path.is_file():
        return {}
    first_line = thermo_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0].strip()
    if not first_line:
        return {}
    header = first_line.lstrip("# ").replace(",", " ").split()
    data = np.atleast_2d(np.genfromtxt(thermo_path, comments="#", dtype=float))
    thermo = {name: data[:, index] for index, name in enumerate(header)}
    if "time_fs" in thermo:
        thermo["time_ps"] = thermo["time_fs"] / FS_PER_PS
    return thermo


def parse_lattice(comment: str) -> np.ndarray:
    match = LATTICE_RE.search(comment)
    if not match:
        raise ValueError(f"Frame comment has no Lattice field: {comment[:120]}")
    values = np.fromstring(match.group(1), sep=" ", dtype=float)
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def iter_xyz_frames(xyz_path: Path) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    with xyz_path.open("r", encoding="utf-8", errors="ignore") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue
            natoms = int(natoms_line)
            comment = handle.readline()
            if not comment:
                raise ValueError(f"{xyz_path}: missing comment line at frame {frame_index}")

            symbols: list[str] = []
            positions = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"{xyz_path}: malformed atom line at frame {frame_index}, atom {atom_index}")
                symbols.append(fields[0])
                positions[atom_index] = [float(fields[1]), float(fields[2]), float(fields[3])]

            yield frame_index, np.array(symbols), positions, parse_lattice(comment)
            frame_index += 1


def minimum_image_distances(h_positions: np.ndarray, o_positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = h_positions[:, None, :] - o_positions[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def classify_frame(symbols: np.ndarray, positions: np.ndarray, cell: np.ndarray, oh_cutoff: float) -> dict[str, int | float | bool]:
    oxygen_indices = np.flatnonzero(symbols == "O")
    hydrogen_indices = np.flatnonzero(symbols == "H")
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        raise ValueError("Frame does not contain both O and H atoms.")

    distances = minimum_image_distances(positions[hydrogen_indices], positions[oxygen_indices], cell)
    nearest_o = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(hydrogen_indices.size), nearest_o]
    bonded = nearest_distances <= oh_cutoff
    attached_h = np.bincount(nearest_o[bonded], minlength=oxygen_indices.size).astype(int)

    counts = {
        "O": int(np.count_nonzero(attached_h == 0)),
        "OH": int(np.count_nonzero(attached_h == 1)),
        "H2O": int(np.count_nonzero(attached_h == 2)),
        "H3O": int(np.count_nonzero(attached_h == 3)),
        "H4O_or_more": int(np.count_nonzero(attached_h >= 4)),
        "unassigned_H": int(np.count_nonzero(~bonded)),
        "min_nearest_OH_distance_A": float(np.min(nearest_distances)),
        "max_bonded_OH_distance_A": float(np.max(nearest_distances[bonded])) if np.any(bonded) else np.nan,
    }
    counts["autoionized"] = bool(counts["OH"] > 0 and counts["H3O"] > 0)
    counts["non_water_oxygens"] = int(counts["O"] + counts["OH"] + counts["H3O"] + counts["H4O_or_more"])
    return counts


def analyze_trajectory(
    xyz_path: Path,
    thermo_path: Path,
    oh_cutoff: float,
    stride: int,
    max_frames: int | None,
) -> list[dict[str, int | float | bool]]:
    thermo = load_thermo(thermo_path)
    thermo_time = thermo.get("time_ps")
    rows: list[dict[str, int | float | bool]] = []

    for frame_index, symbols, positions, cell in iter_xyz_frames(xyz_path):
        if frame_index % stride != 0:
            continue
        if max_frames is not None and len(rows) >= max_frames:
            break

        row = classify_frame(symbols, positions, cell, oh_cutoff)
        row["frame"] = frame_index
        row["time_ps"] = (
            float(thermo_time[frame_index])
            if thermo_time is not None and frame_index < thermo_time.size
            else float(frame_index)
        )
        row["temperature_K"] = (
            float(thermo["temperature_K"][frame_index])
            if "temperature_K" in thermo and frame_index < thermo["temperature_K"].size
            else np.nan
        )
        row["pressure_GPa"] = (
            float(thermo["pressure_GPa"][frame_index])
            if "pressure_GPa" in thermo and frame_index < thermo["pressure_GPa"].size
            else np.nan
        )
        rows.append(row)

    if not rows:
        raise ValueError("No frames were sampled.")
    return rows


def write_csv(rows: list[dict[str, int | float | bool]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "r09_hot_w_water_autoionization_timeseries.csv"
    columns = [
        "frame",
        "time_ps",
        "temperature_K",
        "pressure_GPa",
        *SPECIES,
        "unassigned_H",
        "non_water_oxygens",
        "autoionized",
        "min_nearest_OH_distance_A",
        "max_bonded_OH_distance_A",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_timeseries(rows: list[dict[str, int | float | bool]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "r09_hot_w_water_autoionization_timeseries.png"
    time = np.array([float(row["time_ps"]) for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True, constrained_layout=True)

    colors = {"H2O": "#2166ac", "OH": "#b2182b", "H3O": "#ef8a62", "O": "#4d4d4d", "H4O_or_more": "#7b3294"}
    for species in ("H2O", "OH", "H3O", "O", "H4O_or_more"):
        values = np.array([int(row[species]) for row in rows])
        if species == "H2O" or np.any(values):
            axes[0].step(time, values, where="post", label=species, linewidth=1.2, color=colors[species])

    axes[1].step(time, [int(row["non_water_oxygens"]) for row in rows], where="post", label="non-H2O oxygens", color="#1b7837")
    axes[1].step(time, [int(row["unassigned_H"]) for row in rows], where="post", label="unassigned H", color="#762a83")
    axes[2].plot(time, [float(row["temperature_K"]) for row in rows], linewidth=1.0, color="#d6604d")
    axes[3].plot(time, [float(row["pressure_GPa"]) for row in rows], linewidth=1.0, color="#4393c3")

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("O-centered species count")
    axes[1].set_ylabel("Anomaly count")
    axes[2].set_ylabel("Temperature (K)")
    axes[3].set_ylabel("Pressure (GPa)")
    axes[3].set_xlabel("Time (ps)")
    axes[0].set_title("r09_hot_w water molecular distribution over time")
    axes[1].set_title("Autoionization candidates: OH and H3O in the same frame")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def summarize_events(rows: list[dict[str, int | float | bool]]) -> str:
    events = [row for row in rows if bool(row["autoionized"])]
    if not events:
        return "No sampled frames contained both OH and H3O with the selected cutoff."
    return (
        f"{len(events)} sampled frame(s) contained both OH and H3O. "
        f"First: frame {events[0]['frame']} at {float(events[0]['time_ps']):.6g} ps. "
        f"Last: frame {events[-1]['frame']} at {float(events[-1]['time_ps']):.6g} ps."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot r09_hot_w water autoionization candidates from an XYZ trajectory.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--xyz", type=Path, default=None)
    parser.add_argument("--thermo", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--oh-cutoff", type=float, default=1.25, help="O-H assignment cutoff in Angstrom.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.oh_cutoff <= 0:
        parser.error("--oh-cutoff must be positive")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")

    xyz_path = args.xyz or find_one(args.run_dir, "*.xyz")
    if xyz_path is None:
        raise FileNotFoundError(f"No .xyz trajectory found in {args.run_dir}")
    thermo_path = args.thermo or find_one(args.run_dir, "*thermo*.txt") or Path()
    output_dir = args.output_dir or args.run_dir / "plots"

    rows = analyze_trajectory(xyz_path, thermo_path, args.oh_cutoff, args.stride, args.max_frames)
    csv_path = write_csv(rows, output_dir)
    plot_path = plot_timeseries(rows, output_dir)

    print(f"Analyzed {len(rows)} sampled frame(s) from {xyz_path}")
    print(f"O-H cutoff: {args.oh_cutoff:g} A")
    print(summarize_events(rows))
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
