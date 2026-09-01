#!/usr/bin/env python3
"""DEPRECATED: MACE-POLAR partial-charge prediction workflow.

This script is retained only for historical inspection. Do not use it for new
DFT/MLIP production or validation; current workflows must use formal integer
charges only.

Default input is the 20-frame focused H2-formation cluster trajectory extracted
from the 15 GPa, 300 K r09_hot_w temperature ramp. The script streams frames,
evaluates MACE-POLAR, and writes per-atom charges plus per-frame summaries.
MACE-POLAR charge units are reported as elementary-charge units (e).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

raise SystemExit(
    "DEPRECATED: predictedcharges.py uses MLIP predicted partial charges. "
    "Use formal integer charges only."
)

import numpy as np
from ase import Atoms
from ase.io import iread, write


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "outputsfull" / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputsfull"
    / "temperature_ramp"
    / "r09_hot_w"
    / "plots"
    / "focused.xyz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputsfull" / "7_30b_predictedcharges_focused"
DEFAULT_MODEL = "polar-1-m"
DEFAULT_DTYPE = "float32"
DEFAULT_DEVICE = os.environ.get("MLIP_MACE_DEVICE", "cuda")
DEFAULT_CHARGE = 0
DEFAULT_SPIN = 1
DEFAULT_EXTERNAL_FIELD = (0.0, 0.0, 0.0)


def status(message: str) -> None:
    print(message, flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def external_field(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("must have exactly three comma-separated values, e.g. 0,0,0")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict MACE-POLAR atomic charges for selected frames of an ASE-readable trajectory."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input trajectory readable by ASE.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MACE-POLAR model name/path. Default: {DEFAULT_MODEL}")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="MACE device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default=DEFAULT_DTYPE,
        help=f"MACE default dtype. Default: {DEFAULT_DTYPE}",
    )
    parser.add_argument("--charge", type=int, default=DEFAULT_CHARGE, help="Total system charge passed to MACE-POLAR.")
    parser.add_argument("--spin", type=int, default=DEFAULT_SPIN, help="Spin multiplicity passed to MACE-POLAR.")
    parser.add_argument(
        "--external-field",
        type=external_field,
        default=DEFAULT_EXTERNAL_FIELD,
        help="External electric field as Ex,Ey,Ez. Default: 0,0,0.",
    )
    parser.add_argument("--start", type=nonnegative_int, default=0, help="First zero-based trajectory frame to consider.")
    parser.add_argument("--stride", type=positive_int, default=1, help="Evaluate every Nth considered frame.")
    parser.add_argument("--max-frames", type=positive_int, default=None, help="Maximum number of sampled frames.")
    parser.add_argument(
        "--no-write-extxyz",
        dest="write_extxyz",
        action="store_false",
        help="Do not write predicted.xyz with the per-atom Charge property.",
    )
    parser.add_argument(
        "--no-plot-charges",
        dest="plot_charges",
        action="store_false",
        help="Do not write charge-labeled PNG visualizations for sampled frames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the intended outputs without loading MACE-POLAR.",
    )
    parser.set_defaults(write_extxyz=True, plot_charges=True)
    return parser.parse_args()


def build_calculator(model: str, device: str, dtype: str) -> Any:
    os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR))
    local_mace = REPO_ROOT / "mace"
    if local_mace.is_dir():
        sys.path.insert(0, str(local_mace))

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "MLIP_MACE_DEVICE/--device requests CUDA, but torch.cuda.is_available() is False. "
            "Run on a GPU node or set --device cpu for a CPU check."
        )

    try:
        from mace.calculators import mace_polar
    except ModuleNotFoundError as exc:
        if exc.name == "graph_longrange":
            raise ModuleNotFoundError(
                "MACE-POLAR requires graph_longrange/graph_electrostatics. "
                "Run inside the environment that has the MACE-POLAR dependencies installed."
            ) from exc
        raise

    status(f"Using MACE-POLAR model={model}, device={device}, dtype={dtype}")
    status(f"Using XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}")
    return mace_polar(model=model, device=device, default_dtype=dtype)


def sampled_frames(path: Path, start: int, stride: int, max_frames: int | None):
    if not path.is_file():
        raise FileNotFoundError(f"Input trajectory not found: {path}")

    emitted = 0
    for frame_index, atoms in enumerate(iread(path, index=":")):
        if frame_index < start:
            continue
        if (frame_index - start) % stride != 0:
            continue
        yield frame_index, atoms
        emitted += 1
        if max_frames is not None and emitted >= max_frames:
            break


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def predicted_charge_arrays(calc: Any, atoms: Atoms) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if "charges" in calc.results:
        charges = as_numpy(calc.results["charges"]).reshape(-1)
        if charges.shape != (len(atoms),):
            raise ValueError(f"calc.results['charges'] has shape {charges.shape}; expected ({len(atoms)},)")
        return charges, None, None

    if "spin_charge_density" in calc.results:
        p_spin = as_numpy(calc.results["spin_charge_density"])
        if p_spin.shape[:3] != (len(atoms), 2, 4):
            raise ValueError(
                f"calc.results['spin_charge_density'] has shape {p_spin.shape}; "
                f"expected ({len(atoms)}, 2, 4)"
            )
        charges_up = p_spin[:, 0, 0]
        charges_down = p_spin[:, 1, 0]
        return charges_up + charges_down, charges_up, charges_down

    available = ", ".join(sorted(calc.results)) or "<none>"
    raise KeyError(f"Calculator results contain no charges or spin_charge_density arrays. Available keys: {available}")


def prepare_atoms(
    atoms: Atoms,
    charge: int,
    spin: int,
    field: tuple[float, float, float],
    calc: Any,
) -> Atoms:
    atoms = atoms.copy()
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.info["external_field"] = list(field)
    atoms.calc = calc
    return atoms


def output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    stem = input_path.stem
    atom_csv = output_dir / f"{stem}_mace_polar_predicted_charges.csv"
    frame_csv = output_dir / f"{stem}_mace_polar_charge_summary.csv"
    extxyz = output_dir / "predicted.xyz"
    plot_dir = output_dir / f"{stem}_charge_labeled_frames"
    return atom_csv, frame_csv, extxyz, plot_dir


def element_color(symbol: str) -> str:
    colors = {
        "H": "#f7f7f7",
        "C": "#4d4d4d",
        "N": "#4f6bed",
        "O": "#d62728",
        "S": "#f2c94c",
    }
    return colors.get(symbol, "#9e9e9e")


def covalent_radius(symbol: str) -> float:
    radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
    }
    return radii.get(symbol, 0.75)


def bond_pairs(atoms: Atoms, scale: float = 1.25) -> list[tuple[int, int]]:
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    pairs: list[tuple[int, int]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            cutoff = scale * (covalent_radius(symbols[i]) + covalent_radius(symbols[j]))
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance <= cutoff:
                pairs.append((i, j))
    return pairs


def charge_color_limits(charges: np.ndarray) -> tuple[float, float]:
    max_abs = float(np.max(np.abs(charges))) if charges.size else 1.0
    max_abs = max(max_abs, 1.0e-6)
    return -max_abs, max_abs


def plot_charge_frame(atoms: Atoms, frame_index: int, sample_index: int, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    charges = np.asarray(atoms.arrays["Charge"], dtype=float)
    positions = atoms.positions
    symbols = atoms.get_chemical_symbols()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_index:06d}_sample_{sample_index:04d}_charges.png"

    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    vmin, vmax = charge_color_limits(charges)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, j in bond_pairs(atoms):
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="#9a9a9a",
            linewidth=1.6,
            zorder=1,
        )

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=charges,
        cmap="coolwarm",
        norm=norm,
        s=[210 if symbol != "H" else 155 for symbol in symbols],
        edgecolors="black",
        linewidths=0.9,
        zorder=2,
    )

    for atom_index, (symbol, position, charge) in enumerate(zip(symbols, positions, charges)):
        ax.text(
            float(position[0]),
            float(position[1]) + 0.12,
            f"{symbol}{atom_index}\n{charge:+.3f} e",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            zorder=3,
        )

    for symbol in sorted(set(symbols)):
        idx = symbols.index(symbol)
        ax.scatter(
            [],
            [],
            s=170 if symbol != "H" else 120,
            facecolor=element_color(symbol),
            edgecolor="black",
            label=symbol,
        )

    charge_sum = float(np.sum(charges))
    source_frame = atoms.info.get("source_frame", frame_index)
    source_time = atoms.info.get("source_time_ps", None)
    time_text = f", source time {float(source_time):.4g} ps" if source_time is not None else ""
    ax.set_title(f"MACE-POLAR charges, frame {frame_index} (source frame {source_frame}{time_text})")
    ax.set_xlabel("x (Angstrom)")
    ax.set_ylabel("y (Angstrom)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.legend(title="Element", loc="upper right", frameon=True)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.84, pad=0.02)
    colorbar.set_label("MACE-POLAR atomic charge (e)")

    pad = 1.2
    ax.set_xlim(float(np.min(positions[:, 0]) - pad), float(np.max(positions[:, 0]) + pad))
    ax.set_ylim(float(np.min(positions[:, 1]) - pad), float(np.max(positions[:, 1]) + pad))
    ax.text(
        0.01,
        0.01,
        f"sum q = {charge_sum:+.4e} e",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.86},
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def write_headers(atom_writer: csv.DictWriter[str], frame_writer: csv.DictWriter[str]) -> None:
    atom_writer.writeheader()
    frame_writer.writeheader()


def evaluate_charges(args: argparse.Namespace) -> tuple[int, Path, Path, Path, Path | None]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atom_csv, frame_csv, extxyz_path, plot_dir = output_paths(args.input, args.output_dir)
    calc = build_calculator(args.model, args.device, args.dtype)

    atom_fields = [
        "frame",
        "sample",
        "atom_index",
        "symbol",
        "x_A",
        "y_A",
        "z_A",
        "mace_polar_charge_e",
        "spin_up_charge_e",
        "spin_down_charge_e",
    ]
    frame_fields = [
        "frame",
        "sample",
        "natoms",
        "energy_eV",
        "sum_mace_polar_charge_e",
        "mean_mace_polar_charge_e",
        "min_mace_polar_charge_e",
        "max_mace_polar_charge_e",
        "charge_setting_e",
        "spin_multiplicity",
        "external_field_x",
        "external_field_y",
        "external_field_z",
    ]

    frames_done = 0
    with atom_csv.open("w", encoding="utf-8", newline="") as atom_handle, frame_csv.open(
        "w", encoding="utf-8", newline=""
    ) as frame_handle:
        atom_writer = csv.DictWriter(atom_handle, fieldnames=atom_fields)
        frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields)
        write_headers(atom_writer, frame_writer)

        for sample_index, (frame_index, raw_atoms) in enumerate(
            sampled_frames(args.input, args.start, args.stride, args.max_frames)
        ):
            atoms = prepare_atoms(raw_atoms, args.charge, args.spin, args.external_field, calc)
            energy_ev = float(atoms.get_potential_energy())
            charges, charges_up, charges_down = predicted_charge_arrays(calc, atoms)

            charges = np.asarray(charges, dtype=float)
            atoms.set_array("Charge", charges)
            if charges_up is not None and charges_down is not None:
                atoms.set_array("SpinUpCharge", np.asarray(charges_up, dtype=float))
                atoms.set_array("SpinDownCharge", np.asarray(charges_down, dtype=float))
            atoms.info["mace_polar_energy_eV"] = energy_ev

            for atom_index, (symbol, position, charge_value) in enumerate(
                zip(atoms.get_chemical_symbols(), atoms.positions, charges)
            ):
                atom_writer.writerow(
                    {
                        "frame": frame_index,
                        "sample": sample_index,
                        "atom_index": atom_index,
                        "symbol": symbol,
                        "x_A": f"{float(position[0]):.10g}",
                        "y_A": f"{float(position[1]):.10g}",
                        "z_A": f"{float(position[2]):.10g}",
                        "mace_polar_charge_e": f"{float(charge_value):.12g}",
                        "spin_up_charge_e": ""
                        if charges_up is None
                        else f"{float(charges_up[atom_index]):.12g}",
                        "spin_down_charge_e": ""
                        if charges_down is None
                        else f"{float(charges_down[atom_index]):.12g}",
                    }
                )

            frame_writer.writerow(
                {
                    "frame": frame_index,
                    "sample": sample_index,
                    "natoms": len(atoms),
                    "energy_eV": f"{energy_ev:.12g}",
                    "sum_mace_polar_charge_e": f"{float(np.sum(charges)):.12g}",
                    "mean_mace_polar_charge_e": f"{float(np.mean(charges)):.12g}",
                    "min_mace_polar_charge_e": f"{float(np.min(charges)):.12g}",
                    "max_mace_polar_charge_e": f"{float(np.max(charges)):.12g}",
                    "charge_setting_e": args.charge,
                    "spin_multiplicity": args.spin,
                    "external_field_x": args.external_field[0],
                    "external_field_y": args.external_field[1],
                    "external_field_z": args.external_field[2],
                }
            )

            if args.write_extxyz:
                write(extxyz_path, atoms, format="extxyz", append=frames_done > 0)
            if args.plot_charges:
                plot_charge_frame(atoms, frame_index, sample_index, plot_dir)

            frames_done += 1
            if frames_done == 1 or frames_done % 10 == 0:
                status(f"Evaluated {frames_done} sampled frame(s); latest trajectory frame {frame_index}")

    if frames_done == 0:
        raise ValueError(
            f"No frames selected from {args.input} with start={args.start}, stride={args.stride}, "
            f"max_frames={args.max_frames}"
        )
    return frames_done, atom_csv, frame_csv, extxyz_path, plot_dir if args.plot_charges else None


def main() -> None:
    args = parse_args()
    atom_csv, frame_csv, extxyz_path, plot_dir = output_paths(args.input, args.output_dir)

    if args.dry_run:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input trajectory not found: {args.input}")
        status(f"Input: {args.input}")
        status(f"Model: {args.model}")
        status(f"Device: {args.device}")
        status(f"Frame selection: start={args.start}, stride={args.stride}, max_frames={args.max_frames}")
        status(f"Would write atom charges CSV: {atom_csv}")
        status(f"Would write frame summary CSV: {frame_csv}")
        if args.write_extxyz:
            status(f"Would write annotated extxyz: {extxyz_path}")
        if args.plot_charges:
            status(f"Would write charge-labeled PNG frames under: {plot_dir}")
        return

    frames_done, atom_csv, frame_csv, extxyz_path, plot_dir = evaluate_charges(args)
    status(f"Wrote per-atom charges for {frames_done} sampled frame(s): {atom_csv}")
    status(f"Wrote frame charge summary: {frame_csv}")
    if args.write_extxyz:
        status(f"Wrote annotated trajectory: {extxyz_path}")
    if plot_dir is not None:
        status(f"Wrote charge-labeled frame PNGs under: {plot_dir}")


if __name__ == "__main__":
    main()
