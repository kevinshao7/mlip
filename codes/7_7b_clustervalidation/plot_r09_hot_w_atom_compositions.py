from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "outputsfull" / "temperature_ramp" / "r09_hot_w"
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
SPECIES = ("O", "OH", "H2O", "H3O", "H4O_or_more")


def find_one(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    if pattern == "*.xyz":
        files = [path for path in files if "checkpoint" not in path.name.lower()] or files
    return files[0] if files else None


def parse_lattice(comment: str) -> np.ndarray:
    match = LATTICE_RE.search(comment)
    if not match:
        raise ValueError(f"Frame comment has no Lattice field: {comment[:120]}")
    values = np.fromstring(match.group(1), sep=" ", dtype=float)
    if values.size != 9:
        raise ValueError(f"Expected 9 lattice values, found {values.size}")
    return values.reshape(3, 3)


def iter_xyz_atoms(xyz_path: Path, stride: int, max_frames: int | None) -> Iterator[tuple[int, Atoms]]:
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

            if frame_index % stride == 0:
                yield frame_index, Atoms(symbols=symbols, positions=positions, cell=parse_lattice(comment), pbc=True)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_index += 1


def minimum_image_distances(h_positions: np.ndarray, o_positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = h_positions[:, None, :] - o_positions[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    delta -= np.round(fractional) @ cell
    return np.linalg.norm(delta, axis=2)


def water_species_counts(atoms: Atoms, oh_cutoff: float) -> dict[str, int]:
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.positions
    oxygen_indices = np.flatnonzero(symbols == "O")
    hydrogen_indices = np.flatnonzero(symbols == "H")
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        raise ValueError("Frame does not contain both O and H atoms.")

    distances = minimum_image_distances(positions[hydrogen_indices], positions[oxygen_indices], atoms.cell.array)
    nearest_o = np.argmin(distances, axis=1)
    bonded = distances[np.arange(hydrogen_indices.size), nearest_o] <= oh_cutoff
    attached_h = np.bincount(nearest_o[bonded], minlength=oxygen_indices.size).astype(int)

    return {
        "O": int(np.count_nonzero(attached_h == 0)),
        "OH": int(np.count_nonzero(attached_h == 1)),
        "H2O": int(np.count_nonzero(attached_h == 2)),
        "H3O": int(np.count_nonzero(attached_h == 3)),
        "H4O_or_more": int(np.count_nonzero(attached_h >= 4)),
        "unassigned_H": int(np.count_nonzero(~bonded)),
    }


def composition_label(species_counts: dict[str, int]) -> str:
    parts = [f"{species}({species_counts[species]})" for species in SPECIES if species_counts[species] > 0]
    if species_counts["unassigned_H"] > 0:
        parts.append(f"unassigned_H({species_counts['unassigned_H']})")
    return ":".join(parts) if parts else "none"


def collect_compositions(
    xyz_path: Path,
    stride: int,
    max_frames: int | None,
    oh_cutoff: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_index, atoms in iter_xyz_atoms(xyz_path, stride, max_frames):
        atom_counts = Counter(atoms.get_chemical_symbols())
        species_counts = water_species_counts(atoms, oh_cutoff)
        row: dict[str, object] = {
            "frame": frame_index,
            "natoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "n_water_molecules": species_counts["H2O"],
            "molecular_composition": composition_label(species_counts),
        }
        row.update({f"atom_{symbol}": count for symbol, count in atom_counts.items()})
        row.update(species_counts)
        rows.append(row)
    if not rows:
        raise ValueError("No frames were sampled.")
    return rows


def write_csv(rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    element_columns = sorted({key for row in rows for key in row if key.startswith("atom_")})
    columns = [
        "frame",
        "natoms",
        "formula",
        *element_columns,
        *SPECIES,
        "unassigned_H",
        "n_water_molecules",
        "molecular_composition",
    ]
    output_path = output_dir / "r09_hot_w_atom_compositions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, 0) for column in columns})
    return output_path


def write_composition_counts(rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["molecular_composition"]) for row in rows)
    output_path = output_dir / "r09_hot_w_atom_composition_counts.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["composition", "count", "fraction"])
        writer.writeheader()
        for composition, count in counts.most_common():
            writer.writerow(
                {
                    "composition": composition,
                    "count": count,
                    "fraction": count / len(rows),
                }
            )
    return output_path


def short_label(label: str, max_chars: int = 58) -> str:
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 3]}..."


def plot_composition_distribution(rows: list[dict[str, object]], output_dir: Path, top_n: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["molecular_composition"]) for row in rows)
    common = counts.most_common(top_n)
    labels = [short_label(label) for label, _count in reversed(common)]
    values = np.array([count for _label, count in reversed(common)], dtype=float)
    fractions = 100.0 * values / len(rows)

    fig_height = max(5.5, 0.34 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#4c78a8", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.set_xlabel("Sampled frame count")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.bar_label(
        bars,
        labels=[f"{int(value)} ({fraction:.1f}%)" for value, fraction in zip(values, fractions)],
        padding=3,
        fontsize=8,
    )
    ax.set_xlim(0, max(values) * 1.28)
    omitted = len(counts) - len(common)
    if omitted > 0:
        omitted_frames = len(rows) - sum(count for _label, count in common)
        ax.set_title(
            "r09_hot_w water composition from nearest-O O-H cutoff\n"
            f"Top {len(common)} of {len(counts)} unique compositions; "
            f"{omitted_frames} frame(s) omitted from plot and retained in counts CSV"
        )
    else:
        ax.set_title("r09_hot_w water composition from nearest-O O-H cutoff")

    output_path = output_dir / "r09_hot_w_atom_composition_distribution.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot atom and water composition distributions for r09_hot_w.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--xyz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--oh-cutoff", type=float, default=1.25, help="O-H assignment cutoff in Angstrom.")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.oh_cutoff <= 0:
        parser.error("--oh-cutoff must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")

    xyz_path = args.xyz or find_one(args.run_dir, "*.xyz")
    if xyz_path is None:
        raise FileNotFoundError(f"No .xyz trajectory found in {args.run_dir}")
    output_dir = args.output_dir or args.run_dir / "plots"
    rows = collect_compositions(xyz_path, args.stride, args.max_frames, args.oh_cutoff)
    csv_path = write_csv(rows, output_dir)
    counts_path = write_composition_counts(rows, output_dir)
    plot_path = plot_composition_distribution(rows, output_dir, args.top_n)

    print(f"Analyzed {len(rows)} sampled frame(s) from {xyz_path}")
    print(f"O-H cutoff: {args.oh_cutoff:g} A")
    print(f"Saved {csv_path}")
    print(f"Saved {counts_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
