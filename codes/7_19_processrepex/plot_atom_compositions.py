from __future__ import annotations

import argparse
import sys
import types
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read
from scipy.io import netcdf_file


DEFAULT_INPUT = Path(r"C:\Users\shaoq\Documents\Mainz\mlip\outputsfull\7_20_repex")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "plots"
ASEMOLEC_SOURCE = Path(__file__).resolve().parents[2] / "src" / "asemolec"
DETECTED_PYTHON = Path(r"C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe")


def import_asemolec_ana_atoms():
    """Import aseMolec from the local checkout used by this workspace."""
    if str(ASEMOLEC_SOURCE) not in sys.path:
        sys.path.insert(0, str(ASEMOLEC_SOURCE))

    try:
        from aseMolec import anaAtoms as aa
    except ModuleNotFoundError as exc:
        if exc.name != "ase_ga":
            raise

        ase_ga = types.ModuleType("ase_ga")
        utilities = types.ModuleType("ase_ga.utilities")

        def get_rdf(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "ase-ga is required for aseMolec RDF utilities, but not for "
                "composition plotting. Install it in the detected Python "
                "environment if you call RDF functions."
            )

        utilities.get_rdf = get_rdf
        ase_ga.utilities = utilities
        sys.modules["ase_ga"] = ase_ga
        sys.modules["ase_ga.utilities"] = utilities
        from aseMolec import anaAtoms as aa

    return aa


def replica_number(path: Path) -> int:
    stem = path.name.split("_", 2)
    if len(stem) >= 2 and stem[0] == "replica" and stem[1].isdigit():
        return int(stem[1])
    return 10**9


def load_template_symbols(template_path: Path) -> list[str]:
    if not template_path.is_file():
        raise FileNotFoundError(f"Template structure not found: {template_path}")
    return read(template_path, index=0).get_chemical_symbols()


def iter_netcdf_frames(
    trajectory_path: Path,
    symbols: list[str],
    stride: int,
    max_frames: int | None,
):
    with netcdf_file(trajectory_path, "r", mmap=False) as dataset:
        coordinates = dataset.variables["coordinates"].data
        cell_lengths = dataset.variables["cell_lengths"].data
        cell_angles = dataset.variables["cell_angles"].data
        total_frames = coordinates.shape[0]
        indices = range(0, total_frames, stride)
        if max_frames is not None:
            indices = list(indices)[:max_frames]

        for frame_index in indices:
            cellpar = np.concatenate((cell_lengths[frame_index], cell_angles[frame_index]))
            yield frame_index, Atoms(
                symbols=symbols,
                positions=np.array(coordinates[frame_index], dtype=float),
                cell=cellpar_to_cell(cellpar),
                pbc=True,
            )


def iter_replica_structures(replica_dir: Path, source: str, stride: int, max_frames: int | None):
    if source == "minimized":
        path = replica_dir / "minimized.pdb"
        if not path.is_file():
            return
        yield "minimized.pdb", -1, read(path, index=0)
        return

    template_path = replica_dir / "minimized.pdb"
    trajectory_path = replica_dir / "trajectory.nc"
    if not trajectory_path.is_file():
        return
    symbols = load_template_symbols(template_path)
    for frame_index, atoms in iter_netcdf_frames(trajectory_path, symbols, stride, max_frames):
        yield "trajectory.nc", frame_index, atoms


def formula_counts(atoms: Atoms) -> Counter[str]:
    return Counter(atoms.get_chemical_symbols())


def annotate_composition(atoms: Atoms, bond_scale: float) -> tuple[int, str]:
    aa = import_asemolec_ana_atoms()
    db = [atoms]
    aa.wrap_molecs(db, fct=bond_scale, full=False, prog=False)
    return int(db[0].info["Nmols"]), str(db[0].info["Comp"])


def collect_compositions(
    input_dir: Path,
    source: str,
    stride: int,
    max_frames_per_replica: int | None,
    bond_scale: float,
) -> list[dict[str, object]]:
    replicas = sorted(
        (path for path in input_dir.glob("replica_*") if path.is_dir()),
        key=replica_number,
    )
    if not replicas:
        raise FileNotFoundError(f"No replica_* directories found in {input_dir}")

    rows: list[dict[str, object]] = []
    for replica in replicas:
        for structure_source, frame_index, atoms in iter_replica_structures(
            replica, source, stride, max_frames_per_replica
        ):
            counts = formula_counts(atoms)
            nmols, composition = annotate_composition(atoms, bond_scale)
            row: dict[str, object] = {
                "replica": replica.name,
                "source": structure_source,
                "frame_index": frame_index,
                "natoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "Nmols": nmols,
                "Comp": composition,
            }
            row.update(counts)
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No {source} structures found under replica_* directories in {input_dir}"
        )
    return rows


def plot_composition_distribution(rows: list[dict[str, object]], output_dir: Path, top_n: int) -> Path:
    counts = Counter(str(row["Comp"]) for row in rows)
    common = counts.most_common(top_n)
    if len(common) < len(counts):
        common.append(("Other", sum(counts.values()) - sum(value for _label, value in common)))
    labels, values = zip(*common)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 1.0 else "",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 9},
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(8)
        text.set_color("white")
    ax.set_title("Molecular composition distribution from aseMolec")

    output_path = output_dir / "atom_composition_distribution.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot atom and molecular composition distributions for replica-exchange outputs."
    )
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        choices=("minimized", "trajectory"),
        default="minimized",
        help="Use minimized.pdb structures or sample trajectory.nc files.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Frame stride for --source trajectory.",
    )
    parser.add_argument(
        "--max-frames-per-replica",
        type=int,
        default=None,
        help="Optional cap on sampled trajectory frames per replica.",
    )
    parser.add_argument(
        "--bond-scale",
        type=float,
        default=1.0,
        help="aseMolec neighbor-list natural-cutoff multiplier for molecule detection.",
    )
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_frames_per_replica is not None and args.max_frames_per_replica < 1:
        parser.error("--max-frames-per-replica must be at least 1")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")

    rows = collect_compositions(
        args.input_dir,
        args.source,
        args.stride,
        args.max_frames_per_replica,
        args.bond_scale,
    )
    output = plot_composition_distribution(rows, args.output_dir, args.top_n)

    print(f"Python executable expected for this workflow: {DETECTED_PYTHON}")
    print(f"Python executable used now: {Path(sys.executable)}")
    print(f"Read aseMolec from: {ASEMOLEC_SOURCE}")
    print(f"Analyzed {len(rows)} structure(s).")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
