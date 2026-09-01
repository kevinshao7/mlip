#!/usr/bin/env python3
"""Run MLIP single-point calculations for parity-plot clusters.

Models are selected with --model:
    polar1s  -> MACE-POLAR polar-1-s
    polar1m  -> MACE-POLAR polar-1-m
    off      -> MACE-OFF medium

The default input is the H2-formation cluster trajectory used by the ORCA
parity-plot jobs. Outputs are keyed by source frame so they can be compared
against DFT results later.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread, write


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PARITY_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PARITY_DIR / "7_26_H2pathvalidation" / "r09_hot_w_h2formation_training_clusters.xyz"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputsfull" / "A_parityplot" / "8_6_mlippred"
DEFAULT_CACHE_DIR = REPO_ROOT / "outputsfull" / ".cache"
DEFAULT_DEVICE = os.environ.get("MLIP_MACE_DEVICE", "cuda")
DEFAULT_DTYPE = "float32"
DEFAULT_FRAMES = "0,180"
DEFAULT_SPIN = 1
DEFAULT_EXTERNAL_FIELD = (0.0, 0.0, 0.0)
FORMAL_CHARGES = {"H": 1.0, "N": -3.0, "O": -2.0, "S": -2.0}

MODEL_CONFIGS = {
    "polar1s": {"kind": "polar", "model_name": "polar-1-s", "label": "mace_polar_1s"},
    "polar1m": {"kind": "polar", "model_name": "polar-1-m", "label": "mace_polar_1m"},
    "off": {"kind": "off", "model_name": "medium", "label": "mace_off_medium"},
}


def status(message: str) -> None:
    print(message, flush=True)


def parse_frames(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("must be a half-open range like 0,180")
    start, stop = (int(parts[0]), int(parts[1]))
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= start < stop")
    return start, stop


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def external_field(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("must have three comma-separated values, e.g. 0,0,0")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MACE-POLAR or MACE-OFF single-point predictions on selected cluster frames."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ASE-readable trajectory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), default="polar1s")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="MACE device: cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default=DEFAULT_DTYPE)
    parser.add_argument(
        "--frames",
        type=parse_frames,
        default=parse_frames(DEFAULT_FRAMES),
        help="Half-open frame range start,stop. Default: 0,180.",
    )
    parser.add_argument("--stride", type=positive_int, default=1)
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help=(
            "Override total system charge for MACE-POLAR. By default the charge is computed per frame "
            "from formal atom charges O=-2 and H=+1, matching startup.py/ORCA input generation."
        ),
    )
    parser.add_argument("--spin", type=int, default=DEFAULT_SPIN, help="Spin multiplicity for MACE-POLAR.")
    parser.add_argument("--external-field", type=external_field, default=DEFAULT_EXTERNAL_FIELD)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs for this model.")
    parser.add_argument("--resume", action="store_true", help="Append and skip completed successful frames.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed frames and keep going.")
    parser.add_argument("--no-extxyz", dest="write_extxyz", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without loading MACE.")
    parser.set_defaults(write_extxyz=True)
    return parser.parse_args()


def add_local_mace_to_path() -> None:
    local_mace = REPO_ROOT / "mace"
    if local_mace.is_dir():
        sys.path.insert(0, str(local_mace))


def build_calculator(model_key: str, device: str, dtype: str) -> Any:
    config = MODEL_CONFIGS[model_key]
    os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR))
    add_local_mace_to_path()

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Run on a GPU node or set --device cpu."
        )

    if config["kind"] == "polar":
        try:
            from mace.calculators import mace_polar
        except ModuleNotFoundError as exc:
            if exc.name == "graph_longrange":
                raise ModuleNotFoundError(
                    "MACE-POLAR requires graph_longrange/graph_electrostatics. "
                    "Use the environment that has the MACE-POLAR dependencies installed."
                ) from exc
            raise
        status(f"Using MACE-POLAR model={config['model_name']}, device={device}, dtype={dtype}")
        return mace_polar(model=config["model_name"], device=device, default_dtype=dtype)

    from mace.calculators import mace_off

    status(f"Using MACE-OFF model={config['model_name']}, device={device}, dtype={dtype}")
    try:
        return mace_off(model=config["model_name"], device=device, default_dtype=dtype)
    except TypeError:
        return mace_off(model=config["model_name"], device=device)


def selected_frames(path: Path, start: int, stop: int, stride: int):
    if not path.is_file():
        raise FileNotFoundError(f"Input trajectory not found: {path}")
    for frame_index, atoms in enumerate(iread(path, index=":")):
        if frame_index < start:
            continue
        if frame_index >= stop:
            break
        if (frame_index - start) % stride != 0:
            continue
        yield frame_index, atoms


def formal_charges_for(atoms: Atoms) -> np.ndarray:
    symbols = atoms.get_chemical_symbols()
    unknown = sorted({symbol for symbol in symbols if symbol not in FORMAL_CHARGES})
    if unknown:
        raise KeyError(f"No hard-coded formal charge for elements: {', '.join(unknown)}")
    return np.array([FORMAL_CHARGES[symbol] for symbol in symbols], dtype=float)


def frame_charge_setting(args: argparse.Namespace, formal_charges: np.ndarray) -> int:
    if args.charge is not None:
        return int(args.charge)
    formal_sum = float(np.sum(formal_charges))
    rounded = int(round(formal_sum))
    if not np.isclose(formal_sum, rounded, atol=1.0e-8):
        raise ValueError(f"Formal charge sum is not integral: {formal_sum}")
    return rounded


def prepare_atoms(atoms: Atoms, args: argparse.Namespace, calculator: Any) -> tuple[Atoms, int]:
    atoms = atoms.copy()
    formal_charges = formal_charges_for(atoms)
    charge_setting = frame_charge_setting(args, formal_charges)
    atoms.set_initial_charges(formal_charges)
    atoms.set_array("formal_charge_e", formal_charges)

    if MODEL_CONFIGS[args.model]["kind"] == "polar":
        atoms.info["charge"] = charge_setting
        atoms.info["spin"] = args.spin
        atoms.info["external_field"] = list(args.external_field)

    atoms.calc = calculator
    return atoms, charge_setting


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def force_stats(forces: np.ndarray) -> tuple[float, float]:
    vector_norms = np.linalg.norm(forces, axis=1)
    max_force = float(np.max(vector_norms)) if vector_norms.size else 0.0
    rms_force = float(np.sqrt(np.mean(np.sum(forces**2, axis=1)))) if vector_norms.size else 0.0
    return max_force, rms_force


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    label = MODEL_CONFIGS[args.model]["label"]
    output_dir = args.output_root / args.model
    stem = args.input.stem
    summary_csv = output_dir / f"{stem}_{label}_singlepoints.csv"
    force_csv = output_dir / f"{stem}_{label}_forces.csv"
    extxyz = output_dir / f"{stem}_{label}_predictions.xyz"
    return summary_csv, force_csv, extxyz


def completed_frames(summary_csv: Path) -> set[int]:
    if not summary_csv.is_file():
        return set()
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {int(row["frame"]) for row in reader if row.get("status") == "ok"}


def active_output_paths(args: argparse.Namespace, paths: tuple[Path, Path, Path]) -> tuple[Path, ...]:
    summary_csv, force_csv, extxyz = paths
    if args.write_extxyz:
        return summary_csv, force_csv, extxyz
    return summary_csv, force_csv


def prepare_outputs(args: argparse.Namespace, paths: tuple[Path, Path, Path]) -> set[int]:
    summary_csv, _, _ = paths
    output_dir = summary_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    active_paths = active_output_paths(args, paths)
    existing = [path for path in active_paths if path.exists()]
    if existing and not args.force and not args.resume:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {names}. Use --force or --resume.")

    if args.force:
        for path in existing:
            path.unlink()
        return set()

    return completed_frames(summary_csv) if args.resume else set()


def append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_extxyz(path: Path, atoms: Atoms) -> None:
    atoms = atoms.copy()
    atoms.calc = None
    write(path, atoms, format="extxyz", append=path.exists())


SUMMARY_FIELDS = [
    "frame",
    "model_key",
    "model_name",
    "calculator",
    "device",
    "dtype",
    "natoms",
    "formula",
    "energy_eV",
    "energy_per_atom_eV",
    "max_force_eV_A",
    "rms_force_eV_A",
    "formal_charge_sum_e",
    "mlip_charge_setting_e",
    "status",
    "error",
]

FORCE_FIELDS = [
    "frame",
    "atom_index",
    "symbol",
    "x_A",
    "y_A",
    "z_A",
    "formal_charge_e",
    "force_x_eV_A",
    "force_y_eV_A",
    "force_z_eV_A",
]


def evaluate(args: argparse.Namespace) -> tuple[int, tuple[Path, Path, Path]]:
    summary_csv, force_csv, extxyz = output_paths(args)
    config = MODEL_CONFIGS[args.model]

    if args.dry_run:
        status(f"Input: {args.input}")
        status(f"Model: {args.model} -> {config['model_name']}")
        status(f"Frame range: {args.frames[0]},{args.frames[1]} stride={args.stride}")
        status(f"Output summary: {summary_csv}")
        status(f"Output forces: {force_csv}")
        status(f"Output trajectory: {extxyz}")
        return 0, (summary_csv, force_csv, extxyz)

    done = prepare_outputs(args, (summary_csv, force_csv, extxyz))
    calculator = build_calculator(args.model, args.device, args.dtype)
    evaluated = 0
    for frame_index, raw_atoms in selected_frames(args.input, args.frames[0], args.frames[1], args.stride):
        if frame_index in done:
            status(f"Skipping completed frame {frame_index}")
            continue

        try:
            atoms, charge_setting = prepare_atoms(raw_atoms, args, calculator)
            energy_ev = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            max_force, rms_force = force_stats(forces)
            formal_charges = np.asarray(atoms.arrays["formal_charge_e"], dtype=float)

            atoms.info["mlip_model_key"] = args.model
            atoms.info["mlip_model_name"] = config["model_name"]
            atoms.info["mlip_charge_setting_e"] = charge_setting
            atoms.info["mlip_energy_eV"] = energy_ev
            atoms.arrays["mlip_forces_eV_A"] = forces

            append_row(
                summary_csv,
                SUMMARY_FIELDS,
                {
                    "frame": frame_index,
                    "model_key": args.model,
                    "model_name": config["model_name"],
                    "calculator": config["kind"],
                    "device": args.device,
                    "dtype": args.dtype,
                    "natoms": len(atoms),
                    "formula": atoms.get_chemical_formula(),
                    "energy_eV": f"{energy_ev:.12g}",
                    "energy_per_atom_eV": f"{energy_ev / len(atoms):.12g}",
                    "max_force_eV_A": f"{max_force:.12g}",
                    "rms_force_eV_A": f"{rms_force:.12g}",
                    "formal_charge_sum_e": f"{float(np.sum(formal_charges)):.12g}",
                    "mlip_charge_setting_e": charge_setting,
                    "status": "ok",
                    "error": "",
                },
            )

            symbols = atoms.get_chemical_symbols()
            for atom_index, (symbol, position, formal_charge, force) in enumerate(
                zip(symbols, atoms.positions, formal_charges, forces)
            ):
                append_row(
                    force_csv,
                    FORCE_FIELDS,
                    {
                        "frame": frame_index,
                        "atom_index": atom_index,
                        "symbol": symbol,
                        "x_A": f"{float(position[0]):.10g}",
                        "y_A": f"{float(position[1]):.10g}",
                        "z_A": f"{float(position[2]):.10g}",
                        "formal_charge_e": f"{float(formal_charge):.12g}",
                        "force_x_eV_A": f"{float(force[0]):.12g}",
                        "force_y_eV_A": f"{float(force[1]):.12g}",
                        "force_z_eV_A": f"{float(force[2]):.12g}",
                    },
                )

            if args.write_extxyz:
                append_extxyz(extxyz, atoms)
            evaluated += 1
            status(f"Frame {frame_index}: E={energy_ev:.8f} eV, max|F|={max_force:.6f} eV/A")
        except Exception as exc:
            append_row(
                summary_csv,
                SUMMARY_FIELDS,
                {
                    "frame": frame_index,
                    "model_key": args.model,
                    "model_name": config["model_name"],
                    "calculator": config["kind"],
                    "device": args.device,
                    "dtype": args.dtype,
                    "natoms": len(raw_atoms),
                    "formula": raw_atoms.get_chemical_formula(),
                    "energy_eV": "",
                    "energy_per_atom_eV": "",
                    "max_force_eV_A": "",
                    "rms_force_eV_A": "",
                    "formal_charge_sum_e": "",
                    "mlip_charge_setting_e": "",
                    "status": "error",
                    "error": str(exc),
                },
            )
            if not args.continue_on_error:
                raise
            status(f"Frame {frame_index}: ERROR {exc}")

    return evaluated, (summary_csv, force_csv, extxyz)


def main() -> int:
    args = parse_args()
    evaluated, paths = evaluate(args)
    summary_csv, force_csv, extxyz = paths
    status(f"Evaluated {evaluated} frame(s).")
    status(f"Summary CSV: {summary_csv}")
    status(f"Forces CSV: {force_csv}")
    if args.write_extxyz:
        status(f"Prediction trajectory: {extxyz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
