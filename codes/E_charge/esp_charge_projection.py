#!/usr/bin/env python3
"""Project ORCA and MACE-POLAR ESPs onto the same fitted point-charge model.

This workflow:
1. Reads an ORCA electrostatic-potential cube, or generates one with ``orca_plot``.
2. Uses the cube grid points as the shared probe set after a minimum atom-probe
   distance filter.
3. Evaluates the MACE-POLAR electrostatic potential on the same Cartesian probe
   points from ``density_coefficients`` using the displaced-charge real-space
   construction used in ``graph_longrange`` for l<=1 multipoles.
4. Fits atom-centred point charges independently to the ORCA and MACE ESPs with
   the same AmberTools RESP wrapper workflow through ``py_resp.py``.

The intended use is isolated clusters or molecules with ``pbc = F F F``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.cube import read_cube
from scipy import special


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
ORCA_POTENTIAL_AU_TO_VOLT = HARTREE_TO_EV
VOLT_TO_ORCA_POTENTIAL_AU = 1.0 / ORCA_POTENTIAL_AU_TO_VOLT
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
GRAPH_LONGRANGE_FIELD_CONSTANT = 1.0 / (5.526349406e-3)
COULOMB_PREFactor_VOLT_ANG_PER_E = GRAPH_LONGRANGE_FIELD_CONSTANT / (4.0 * math.pi)
ORCA_XYZ_HEADER_RE = re.compile(r"^\s*\*+\s*xyz(?:file)?\s+(-?\d+)\s+(\d+)\b", re.IGNORECASE)
ORCA_DENSITY_NAME_RE = re.compile(r"^\s*\d+\s*:\s*(\S+)\s*$")
QOUT_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CACHE_DIR = REPO_ROOT / "outputsfull" / ".cache"


@dataclass(frozen=True)
class ProbeSet:
    points_A: np.ndarray
    values_volt: np.ndarray
    min_atom_distance_A: np.ndarray
    selected_cutoff_A: float
    available_count: int
    selected_count: int
    sampled: bool


@dataclass(frozen=True)
class FitResult:
    charges_e: np.ndarray
    rmse_volt: float
    mae_volt: float
    max_abs_error_volt: float
    residuals_volt: np.ndarray


def status(message: str) -> None:
    print(message, flush=True)


def csv_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("fragment output path must end in .csv")
    return path


def parse_float_list(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return [float(part) for part in parts]


def parse_fragment(value: str) -> tuple[str, list[int]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("fragment must look like name:1,2,3")
    name, indices_text = value.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("fragment name cannot be empty")
    indices = [int(part.strip()) for part in indices_text.split(",") if part.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("fragment must include at least one 1-based atom index")
    if min(indices) < 1:
        raise argparse.ArgumentTypeError("fragment indices must be 1-based and positive")
    return name, indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ORCA DFT and MACE-POLAR by projecting both ESPs onto the same "
            "atom-centred point-charge model."
        )
    )
    parser.add_argument("--orca-cube", type=Path, default=None, help="Existing ORCA ESP cube.")
    parser.add_argument(
        "--orca-prefix",
        type=Path,
        default=None,
        help=(
            "ORCA basename without extension, or a .gbw/.densities/.inp path. "
            "Used for cube generation and metadata discovery."
        ),
    )
    parser.add_argument(
        "--generate-orca-cube",
        action="store_true",
        help="Generate the ORCA ESP cube with orca_plot when --orca-cube is absent.",
    )
    parser.add_argument(
        "--orca-density-name",
        default=None,
        help="Density container name for orca_plot, e.g. myjob_scf.scfp.",
    )
    parser.add_argument("--orca-plot-exe", default="orca_plot")
    parser.add_argument(
        "--orca-potential-unit",
        choices=("au", "volt"),
        default="au",
        help="Unit stored in the ORCA ESP cube. Default assumes atomic units and converts to volts.",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=None,
        help="Optional ASE-readable structure. If omitted, the cube geometry is used.",
    )
    parser.add_argument(
        "--orca-input",
        type=Path,
        default=None,
        help="Optional ORCA input file to parse total charge and spin multiplicity.",
    )
    parser.add_argument(
        "--mace-model",
        default="polar-1-m",
        help="PolarMACE model name or explicit .model checkpoint path.",
    )
    parser.add_argument("--device", default=os.environ.get("MLIP_MACE_DEVICE", "cpu"))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--charge", type=int, default=None, help="Total charge in e.")
    parser.add_argument("--spin", type=int, default=None, help="Spin multiplicity.")
    parser.add_argument(
        "--external-field",
        type=parse_float_list,
        default=[0.0, 0.0, 0.0],
        help="Comma-separated external field in the MACE input, default 0,0,0.",
    )
    parser.add_argument(
        "--min-probe-distance-A",
        type=float,
        default=1.2,
        help="Minimum atom-probe distance in Angstrom for the main fit.",
    )
    parser.add_argument(
        "--sensitivity-cutoffs-A",
        type=parse_float_list,
        default=[1.0, 1.2, 1.5, 1.8],
        help="Comma-separated minimum atom-probe distances in Angstrom for sensitivity checks.",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=50000,
        help="Maximum number of probe points after filtering. Deterministic downsampling is used if needed.",
    )
    parser.add_argument(
        "--fragment",
        action="append",
        default=[],
        type=parse_fragment,
        help="Optional fragment charge sum, e.g. water:1,2,3 . Can be repeated.",
    )
    parser.add_argument(
        "--pyresp-exe",
        default="py_resp.py",
        help="PyRESP executable or script path. Default: py_resp.py",
    )
    parser.add_argument(
        "--resp-qwt",
        type=float,
        default=0.0005,
        help="RESP restraint weight passed to py_resp.py via the generated input.",
    )
    parser.add_argument(
        "--resp-free-hydrogens",
        type=int,
        choices=(0, 1),
        default=1,
        help="Set ihfree in the generated RESP input. Default: 1.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_orca_prefix(or_path: Path) -> Path:
    if or_path.suffix.lower() in {".gbw", ".densities", ".inp", ".out", ".cube"}:
        return or_path.with_suffix("")
    return or_path


def unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def candidate_search_roots(args: argparse.Namespace) -> list[Path]:
    roots = [
        Path.cwd(),
        SCRIPT_DIR,
        args.output_dir,
    ]
    if args.orca_input is not None:
        roots.append(args.orca_input.parent)
    return unique_existing_paths(roots)


def discover_existing_orca_cube(args: argparse.Namespace) -> Path | None:
    cubes: list[Path] = []
    for root in candidate_search_roots(args):
        cubes.extend(sorted(root.glob("*.esp.cube")))
        cubes.extend(sorted(root.glob("*.cube")))
    unique_cubes = unique_existing_paths(cubes)
    if len(unique_cubes) == 1:
        return unique_cubes[0]
    return None


def discover_orca_prefix(args: argparse.Namespace) -> Path | None:
    prefixes: list[Path] = []
    for root in candidate_search_roots(args):
        for gbw_path in sorted(root.glob("*.gbw")):
            prefix = gbw_path.with_suffix("")
            if prefix.with_suffix(".densities").is_file():
                prefixes.append(prefix)
    unique_prefixes = unique_existing_paths(prefixes)
    if len(unique_prefixes) == 1:
        return unique_prefixes[0]
    return None


def format_discovery_error(args: argparse.Namespace) -> str:
    lines = ["Provide --orca-cube or use --generate-orca-cube with --orca-prefix."]
    lines.append("Auto-discovery checked:")
    for root in candidate_search_roots(args):
        cube_count = len(list(root.glob("*.cube")))
        prefix_count = sum(1 for gbw_path in root.glob("*.gbw") if gbw_path.with_suffix(".densities").is_file())
        lines.append(
            f"  {root}: cubes={cube_count}, gbw+density prefixes={prefix_count}"
        )
    return "\n".join(lines)


def parse_charge_spin_from_orca_input(path: Path) -> tuple[int, int] | None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        match = ORCA_XYZ_HEADER_RE.match(line.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def resolve_charge_and_spin(args: argparse.Namespace, structure: Atoms) -> tuple[int, int]:
    charge = args.charge
    spin = args.spin

    candidates: list[Path] = []
    if args.orca_input is not None:
        candidates.append(args.orca_input)
    if args.orca_prefix is not None:
        prefix = normalize_orca_prefix(args.orca_prefix)
        candidates.append(prefix.with_suffix(".inp"))

    for candidate in candidates:
        if not candidate.is_file():
            continue
        parsed = parse_charge_spin_from_orca_input(candidate)
        if parsed is None:
            continue
        parsed_charge, parsed_spin = parsed
        if charge is None:
            charge = parsed_charge
        if spin is None:
            spin = parsed_spin
        break

    if charge is None:
        info_charge = structure.info.get("charge")
        if info_charge is not None:
            charge = int(info_charge)
    if spin is None:
        info_spin = structure.info.get("spin")
        if info_spin is not None:
            spin = int(info_spin)

    if charge is None or spin is None:
        raise ValueError(
            "Total charge and spin multiplicity must be provided explicitly or discoverable "
            "from --orca-input/--orca-prefix .inp metadata."
        )
    return charge, spin


def discover_density_name(orca_plot_exe: str, densities_path: Path) -> str:
    command = [orca_plot_exe, str(densities_path)]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(densities_path.parent),
    )
    names = [match.group(1) for match in map(ORCA_DENSITY_NAME_RE.match, result.stdout.splitlines()) if match]
    if len(names) != 1:
        raise RuntimeError(
            f"Could not uniquely determine density name from {densities_path}. "
            f"Found {len(names)} candidates; pass --orca-density-name explicitly."
        )
    return names[0]


def generate_orca_cube(args: argparse.Namespace) -> tuple[Path, str]:
    if args.orca_prefix is None:
        raise ValueError("--generate-orca-cube requires --orca-prefix")
    prefix = normalize_orca_prefix(args.orca_prefix)
    gbw_path = prefix.with_suffix(".gbw")
    densities_path = prefix.with_suffix(".densities")
    if not gbw_path.is_file():
        raise FileNotFoundError(f"Missing ORCA gbw file for orca_plot: {gbw_path}")
    if not densities_path.is_file():
        raise FileNotFoundError(f"Missing ORCA densities file for orca_plot: {densities_path}")

    density_name = args.orca_density_name
    if density_name is None:
        density_name = discover_density_name(args.orca_plot_exe, densities_path)

    status(f"Generating ORCA ESP cube with density {density_name}")
    interactive_input = f"1\n43\n{density_name}\n11\n"
    subprocess.run(
        [args.orca_plot_exe, str(gbw_path), "-i"],
        input=interactive_input,
        check=True,
        text=True,
        cwd=str(prefix.parent),
    )

    cube_path = prefix.parent / f"{density_name}.esp.cube"
    if not cube_path.is_file():
        raise FileNotFoundError(
            f"orca_plot completed but expected cube was not found: {cube_path}"
        )
    return cube_path, density_name


def ensure_orca_cube(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.orca_cube is not None:
        if not args.orca_cube.is_file():
            raise FileNotFoundError(f"ORCA cube not found: {args.orca_cube}")
        return args.orca_cube, None
    discovered_cube = discover_existing_orca_cube(args)
    if discovered_cube is not None:
        status(f"Auto-discovered ORCA cube: {discovered_cube}")
        return discovered_cube, None
    if args.orca_prefix is None:
        discovered_prefix = discover_orca_prefix(args)
        if discovered_prefix is not None:
            args.orca_prefix = discovered_prefix
            args.generate_orca_cube = True
            status(f"Auto-discovered ORCA prefix for cube generation: {discovered_prefix}")
    if args.generate_orca_cube:
        return generate_orca_cube(args)
    raise ValueError(format_discovery_error(args))


def read_orca_cube(path: Path) -> tuple[Atoms, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        cube = read_cube(handle, read_data=True)
    atoms = cube["atoms"]
    data = np.asarray(cube["data"], dtype=float)
    origin_A = np.asarray(cube["origin"], dtype=float)
    spacing_A = np.asarray(cube["spacing"], dtype=float)
    return atoms, data, origin_A, spacing_A


def cube_points_from_origin_and_spacing(
    shape: tuple[int, int, int],
    origin_A: np.ndarray,
    spacing_A: np.ndarray,
) -> np.ndarray:
    grid = np.indices(shape, dtype=float).reshape(3, -1).T
    return origin_A[None, :] + grid @ spacing_A


def validate_structure(reference: Atoms, candidate: Atoms) -> Atoms:
    if len(reference) != len(candidate):
        raise ValueError(
            f"Structure atom count mismatch: cube has {len(reference)}, structure has {len(candidate)}"
        )
    ref_symbols = reference.get_chemical_symbols()
    cand_symbols = candidate.get_chemical_symbols()
    if ref_symbols != cand_symbols:
        raise ValueError("Structure symbols do not match the ORCA cube geometry")
    displacement = np.max(np.abs(reference.positions - candidate.positions))
    if displacement > 1.0e-4:
        raise ValueError(
            f"Structure positions do not match the ORCA cube geometry; max |delta| = {displacement:.3e} A"
        )
    return candidate.copy()


def load_structure(args: argparse.Namespace, cube_atoms: Atoms) -> Atoms:
    if args.structure is None:
        return cube_atoms.copy()
    structure = read(args.structure)
    return validate_structure(cube_atoms, structure)


def dft_values_to_volt(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "volt":
        return values.copy()
    if unit == "au":
        return values * ORCA_POTENTIAL_AU_TO_VOLT
    raise ValueError(f"Unsupported ORCA potential unit: {unit}")


def volt_to_au(values_volt: np.ndarray) -> np.ndarray:
    return np.asarray(values_volt, dtype=float) * VOLT_TO_ORCA_POTENTIAL_AU


def minimum_atom_probe_distances(points_A: np.ndarray, atom_positions_A: np.ndarray) -> np.ndarray:
    deltas = points_A[:, None, :] - atom_positions_A[None, :, :]
    return np.linalg.norm(deltas, axis=2).min(axis=1)


def deterministic_downsample_indices(count: int, target: int) -> np.ndarray:
    if count <= target:
        return np.arange(count, dtype=int)
    return np.linspace(0, count - 1, num=target, dtype=int)


def build_probe_set(
    cube_data: np.ndarray,
    origin_A: np.ndarray,
    spacing_A: np.ndarray,
    atom_positions_A: np.ndarray,
    orca_potential_unit: str,
    min_probe_distance_A: float,
    max_probes: int,
) -> ProbeSet:
    points_A = cube_points_from_origin_and_spacing(cube_data.shape, origin_A, spacing_A)
    values_volt = dft_values_to_volt(cube_data.reshape(-1), orca_potential_unit)
    finite_mask = np.isfinite(values_volt)
    points_A = points_A[finite_mask]
    values_volt = values_volt[finite_mask]

    min_dist_A = minimum_atom_probe_distances(points_A, atom_positions_A)
    distance_mask = min_dist_A >= min_probe_distance_A
    filtered_points_A = points_A[distance_mask]
    filtered_values_volt = values_volt[distance_mask]
    filtered_min_dist_A = min_dist_A[distance_mask]
    if filtered_points_A.size == 0:
        raise ValueError(
            f"No cube probe points survived the minimum atom-probe distance cutoff of {min_probe_distance_A:.3f} A"
        )

    keep = deterministic_downsample_indices(len(filtered_points_A), max_probes)
    sampled = len(keep) != len(filtered_points_A)
    return ProbeSet(
        points_A=filtered_points_A[keep],
        values_volt=filtered_values_volt[keep],
        min_atom_distance_A=filtered_min_dist_A[keep],
        selected_cutoff_A=min_probe_distance_A,
        available_count=len(filtered_points_A),
        selected_count=len(keep),
        sampled=sampled,
    )


def point_charge_design_matrix_volt(points_A: np.ndarray, atom_positions_A: np.ndarray) -> np.ndarray:
    displacements = points_A[:, None, :] - atom_positions_A[None, :, :]
    distances_A = np.linalg.norm(displacements, axis=2)
    if np.any(distances_A <= 0.0):
        raise ValueError("Probe set contains an atom position; cannot build point-charge design matrix")
    return COULOMB_PREFactor_VOLT_ANG_PER_E / distances_A


def point_charge_potential_volt(
    charges_e: np.ndarray,
    points_A: np.ndarray,
    atom_positions_A: np.ndarray,
) -> np.ndarray:
    return point_charge_design_matrix_volt(points_A, atom_positions_A) @ charges_e


def prepare_python_runtime() -> None:
    os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if str(REPO_ROOT / "mace") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "mace"))


def write_pyresp_input(
    path: Path,
    title: str,
    total_charge_e: int,
    atoms: Atoms,
    qwt: float,
    free_hydrogens: int,
) -> None:
    lines = [
        title,
        " &cntrl",
        "  nmol = 1,",
        "  iqopt = 1,",
        f"  ihfree = {int(free_hydrogens)},",
        "  irstrnt = 1,",
        f"  qwt = {qwt:.8f},",
        " /",
        "    1.0",
        title[:80],
        f"{int(total_charge_e):5d}{len(atoms):5d}",
    ]
    for atomic_number in atoms.numbers:
        lines.append(f"{int(atomic_number):5d}{0:5d}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pyresp_espot(
    path: Path,
    atoms: Atoms,
    probe_points_A: np.ndarray,
    potentials_au: np.ndarray,
) -> None:
    atom_positions_bohr = atoms.positions * ANGSTROM_TO_BOHR
    probe_points_bohr = probe_points_A * ANGSTROM_TO_BOHR
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"{len(atoms):5d}{len(probe_points_A):6d}\n")
        for atom_index, (atomic_number, position_bohr) in enumerate(
            zip(atoms.numbers, atom_positions_bohr), start=1
        ):
            handle.write(
                f"{position_bohr[0]:16.7E}{position_bohr[1]:16.7E}{position_bohr[2]:16.7E}"
                f"{int(atomic_number):5d}{atom_index:5d}\n"
            )
        for potential_au, point_bohr in zip(potentials_au, probe_points_bohr):
            handle.write(
                f"{potential_au:16.7E}{point_bohr[0]:16.7E}{point_bohr[1]:16.7E}{point_bohr[2]:16.7E}\n"
            )


def parse_pyresp_qout(path: Path, natoms: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_atom_charge_block = False
    charges: list[float] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%FLAG"):
            in_atom_charge_block = stripped.upper().startswith("%FLAG ATOM CHRG")
            continue
        if not in_atom_charge_block or not stripped or stripped.startswith("%FORMAT"):
            continue
        floats = QOUT_FLOAT_RE.findall(stripped)
        if not floats:
            continue
        charges.append(float(floats[-1]))
        if len(charges) == natoms:
            break
    if len(charges) != natoms:
        raise RuntimeError(
            f"Failed to parse {natoms} charges from PyRESP qout file {path}; parsed {len(charges)}"
        )
    return np.asarray(charges, dtype=float)


def fit_with_pyresp(
    pyresp_exe: str,
    run_dir: Path,
    title: str,
    atoms: Atoms,
    probe_points_A: np.ndarray,
    target_volt: np.ndarray,
    total_charge_e: int,
    qwt: float,
    free_hydrogens: int,
) -> FitResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "fit.respin"
    espot_path = run_dir / "fit.espot"
    output_path = run_dir / "fit.respout"
    qout_path = run_dir / "fit.qout"
    esout_path = run_dir / "fit.esout"

    write_pyresp_input(
        path=input_path,
        title=title,
        total_charge_e=total_charge_e,
        atoms=atoms,
        qwt=qwt,
        free_hydrogens=free_hydrogens,
    )
    write_pyresp_espot(
        path=espot_path,
        atoms=atoms,
        probe_points_A=probe_points_A,
        potentials_au=volt_to_au(target_volt),
    )
    subprocess.run(
        [
            pyresp_exe,
            "-O",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-t",
            str(qout_path),
            "-e",
            str(espot_path),
            "-s",
            str(esout_path),
        ],
        check=True,
        cwd=str(run_dir),
    )
    charges_e = parse_pyresp_qout(qout_path, len(atoms))
    residuals_volt = point_charge_potential_volt(
        charges_e=charges_e,
        points_A=probe_points_A,
        atom_positions_A=atoms.positions,
    ) - target_volt
    return FitResult(
        charges_e=charges_e,
        rmse_volt=float(np.sqrt(np.mean(residuals_volt * residuals_volt))),
        mae_volt=float(np.mean(np.abs(residuals_volt))),
        max_abs_error_volt=float(np.max(np.abs(residuals_volt))),
        residuals_volt=residuals_volt,
    )


def build_mace_calculator(model: str, device: str, dtype: str) -> Any:
    prepare_python_runtime()
    from torch.serialization import add_safe_globals

    add_safe_globals([slice])

    from mace.calculators import mace_polar

    model_arg: str | Path
    model_path = Path(model)
    if model_path.is_file():
        model_arg = model_path
    else:
        model_arg = model
    return mace_polar(model=model_arg, device=device, default_dtype=dtype)


def density_coefficients_from_calculator(calculator: Any, atoms: Atoms) -> np.ndarray:
    atoms = atoms.copy()
    atoms.calc = calculator
    _ = atoms.get_potential_energy()
    if "density_coefficients" not in calculator.results:
        raise RuntimeError("MACE calculator did not produce density_coefficients")
    density = calculator.results["density_coefficients"]
    if hasattr(density, "detach"):
        density = density.detach().cpu().numpy()
    density = np.asarray(density, dtype=float)
    if density.shape != (len(atoms), 4):
        raise ValueError(
            f"density_coefficients shape {density.shape}; expected ({len(atoms)}, 4)"
        )
    return density


def configure_mace_atoms(atoms: Atoms, total_charge_e: int, spin_multiplicity: int, external_field: list[float]) -> Atoms:
    configured = atoms.copy()
    configured.info["charge"] = int(total_charge_e)
    configured.info["spin"] = int(spin_multiplicity)
    configured.info["external_field"] = [float(component) for component in external_field]
    return configured


def displaced_graph_longrange_charges(
    density_coefficients: np.ndarray,
    atom_positions_A: np.ndarray,
    offset_A: float,
) -> tuple[np.ndarray, np.ndarray]:
    natoms = density_coefficients.shape[0]
    positions = np.repeat(atom_positions_A, 4, axis=0).copy()
    charges = np.zeros(natoms * 4, dtype=float)

    x_offsets = np.zeros((natoms, 3), dtype=float)
    y_offsets = np.zeros((natoms, 3), dtype=float)
    z_offsets = np.zeros((natoms, 3), dtype=float)
    x_offsets[:, 0] = offset_A
    y_offsets[:, 1] = offset_A
    z_offsets[:, 2] = offset_A

    positions[1::4] += x_offsets
    positions[2::4] += y_offsets
    positions[3::4] += z_offsets

    charges[1::4] = density_coefficients[:, 3] / offset_A
    charges[2::4] = density_coefficients[:, 1] / offset_A
    charges[3::4] = density_coefficients[:, 2] / offset_A
    charges[0::4] = density_coefficients[:, 0] - (
        charges[1::4] + charges[2::4] + charges[3::4]
    )
    return positions, charges


def evaluate_mace_esp_volt(
    probe_points_A: np.ndarray,
    density_coefficients: np.ndarray,
    atom_positions_A: np.ndarray,
    density_smearing_width_A: float,
    displaced_charge_offset_A: float,
) -> np.ndarray:
    source_positions_A, source_charges_e = displaced_graph_longrange_charges(
        density_coefficients=density_coefficients,
        atom_positions_A=atom_positions_A,
        offset_A=displaced_charge_offset_A,
    )
    deltas = probe_points_A[:, None, :] - source_positions_A[None, :, :]
    distances_A = np.linalg.norm(deltas, axis=2)
    smooth_reciprocal = special.erf(0.5 * distances_A / density_smearing_width_A) / (
        distances_A + 1.0e-6
    )
    return COULOMB_PREFactor_VOLT_ANG_PER_E * (smooth_reciprocal @ source_charges_e)


def mace_descriptor_parameters(calculator: Any) -> tuple[float, float]:
    model = calculator.models[0]
    descriptor = getattr(model, "electric_potential_descriptor", None)
    if descriptor is None:
        raise RuntimeError("Loaded MACE model does not expose electric_potential_descriptor")
    density_smearing_width_A = float(model.atomic_multipoles_smearing_width)
    realspace_features = getattr(descriptor, "realspace_features", None)
    if realspace_features is None:
        raise RuntimeError("MACE electric_potential_descriptor has no realspace_features block")
    displaced_charge_offset_A = float(realspace_features.offset)
    return density_smearing_width_A, displaced_charge_offset_A


def sensitivity_rows(
    cutoffs_A: list[float],
    cube_data: np.ndarray,
    origin_A: np.ndarray,
    spacing_A: np.ndarray,
    atoms: Atoms,
    dft_potential_unit: str,
    max_probes: int,
    mace_esp_full_volt: np.ndarray | None,
    density_coefficients: np.ndarray,
    density_smearing_width_A: float,
    displaced_charge_offset_A: float,
    total_charge_e: int,
    pyresp_exe: str,
    pyresp_root_dir: Path,
    resp_qwt: float,
    resp_free_hydrogens: int,
) -> list[dict[str, object]]:
    points_full_A = cube_points_from_origin_and_spacing(cube_data.shape, origin_A, spacing_A)
    dft_full_volt = dft_values_to_volt(cube_data.reshape(-1), dft_potential_unit)
    finite_mask = np.isfinite(dft_full_volt)
    points_full_A = points_full_A[finite_mask]
    dft_full_volt = dft_full_volt[finite_mask]
    min_dist_full_A = minimum_atom_probe_distances(points_full_A, atoms.positions)
    if mace_esp_full_volt is None:
        mace_esp_full_volt = evaluate_mace_esp_volt(
            probe_points_A=points_full_A,
            density_coefficients=density_coefficients,
            atom_positions_A=atoms.positions,
            density_smearing_width_A=density_smearing_width_A,
            displaced_charge_offset_A=displaced_charge_offset_A,
        )

    rows: list[dict[str, object]] = []
    for cutoff_A in sorted(set(cutoffs_A)):
        mask = min_dist_full_A >= cutoff_A
        if not np.any(mask):
            rows.append(
                {
                    "min_probe_distance_A": cutoff_A,
                    "status": "no_points",
                    "available_points": 0,
                    "selected_points": 0,
                }
            )
            continue
        keep = deterministic_downsample_indices(int(mask.sum()), max_probes)
        selected_points_A = points_full_A[mask][keep]
        selected_dft_volt = dft_full_volt[mask][keep]
        selected_mace_volt = mace_esp_full_volt[mask][keep]
        dft_fit = fit_with_pyresp(
            pyresp_exe=pyresp_exe,
            run_dir=pyresp_root_dir / f"cutoff_{cutoff_A:.2f}_dft".replace(".", "p"),
            title=f"DFT RESP cutoff {cutoff_A:.2f} A",
            atoms=atoms,
            probe_points_A=selected_points_A,
            target_volt=selected_dft_volt,
            total_charge_e=total_charge_e,
            qwt=resp_qwt,
            free_hydrogens=resp_free_hydrogens,
        )
        mace_fit = fit_with_pyresp(
            pyresp_exe=pyresp_exe,
            run_dir=pyresp_root_dir / f"cutoff_{cutoff_A:.2f}_mace".replace(".", "p"),
            title=f"MACE RESP cutoff {cutoff_A:.2f} A",
            atoms=atoms,
            probe_points_A=selected_points_A,
            target_volt=selected_mace_volt,
            total_charge_e=total_charge_e,
            qwt=resp_qwt,
            free_hydrogens=resp_free_hydrogens,
        )
        rows.append(
            {
                "min_probe_distance_A": cutoff_A,
                "status": "ok",
                "available_points": int(mask.sum()),
                "selected_points": len(selected_points_A),
                "dft_rmse_volt": dft_fit.rmse_volt,
                "mace_rmse_volt": mace_fit.rmse_volt,
                "dft_charge_sum_e": float(np.sum(dft_fit.charges_e)),
                "mace_charge_sum_e": float(np.sum(mace_fit.charges_e)),
                "l2_charge_delta_e": float(np.linalg.norm(mace_fit.charges_e - dft_fit.charges_e)),
            }
        )
    return rows


def write_atomic_charges_csv(
    path: Path,
    atoms: Atoms,
    dft_fit: FitResult,
    mace_fit: FitResult,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "atom_index",
                "symbol",
                "x_A",
                "y_A",
                "z_A",
                "dft_resp_charge_e",
                "mace_resp_charge_e",
                "charge_delta_e",
            ],
        )
        writer.writeheader()
        for atom_index, (symbol, position) in enumerate(
            zip(atoms.get_chemical_symbols(), atoms.positions), start=1
        ):
            writer.writerow(
                {
                    "atom_index": atom_index,
                    "symbol": symbol,
                    "x_A": f"{position[0]:.12g}",
                    "y_A": f"{position[1]:.12g}",
                    "z_A": f"{position[2]:.12g}",
                    "dft_resp_charge_e": f"{dft_fit.charges_e[atom_index - 1]:.12g}",
                    "mace_resp_charge_e": f"{mace_fit.charges_e[atom_index - 1]:.12g}",
                    "charge_delta_e": f"{(mace_fit.charges_e[atom_index - 1] - dft_fit.charges_e[atom_index - 1]):.12g}",
                }
            )


def write_metrics_csv(
    path: Path,
    dft_fit: FitResult,
    mace_fit: FitResult,
    probe_set: ProbeSet,
    total_charge_e: int,
    spin_multiplicity: int,
    density_smearing_width_A: float,
    displaced_charge_offset_A: float,
    orca_potential_unit: str,
    resp_qwt: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "charge_e",
                "spin_multiplicity",
                "orca_potential_unit_input",
                "probe_distance_cutoff_A",
                "available_probe_points",
                "selected_probe_points",
                "probe_points_downsampled",
                "fit_backend",
                "resp_qwt",
                "dft_rmse_volt",
                "dft_mae_volt",
                "dft_max_abs_error_volt",
                "mace_rmse_volt",
                "mace_mae_volt",
                "mace_max_abs_error_volt",
                "dft_charge_sum_e",
                "mace_charge_sum_e",
                "mace_density_smearing_width_A",
                "mace_displaced_charge_offset_A",
                "charge_l2_delta_e",
                "charge_max_abs_delta_e",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "charge_e": total_charge_e,
                "spin_multiplicity": spin_multiplicity,
                "orca_potential_unit_input": orca_potential_unit,
                "probe_distance_cutoff_A": f"{probe_set.selected_cutoff_A:.12g}",
                "available_probe_points": probe_set.available_count,
                "selected_probe_points": probe_set.selected_count,
                "probe_points_downsampled": str(probe_set.sampled).lower(),
                "fit_backend": "py_resp.py",
                "resp_qwt": f"{resp_qwt:.12g}",
                "dft_rmse_volt": f"{dft_fit.rmse_volt:.12g}",
                "dft_mae_volt": f"{dft_fit.mae_volt:.12g}",
                "dft_max_abs_error_volt": f"{dft_fit.max_abs_error_volt:.12g}",
                "mace_rmse_volt": f"{mace_fit.rmse_volt:.12g}",
                "mace_mae_volt": f"{mace_fit.mae_volt:.12g}",
                "mace_max_abs_error_volt": f"{mace_fit.max_abs_error_volt:.12g}",
                "dft_charge_sum_e": f"{float(np.sum(dft_fit.charges_e)):.12g}",
                "mace_charge_sum_e": f"{float(np.sum(mace_fit.charges_e)):.12g}",
                "mace_density_smearing_width_A": f"{density_smearing_width_A:.12g}",
                "mace_displaced_charge_offset_A": f"{displaced_charge_offset_A:.12g}",
                "charge_l2_delta_e": f"{float(np.linalg.norm(mace_fit.charges_e - dft_fit.charges_e)):.12g}",
                "charge_max_abs_delta_e": f"{float(np.max(np.abs(mace_fit.charges_e - dft_fit.charges_e))):.12g}",
            }
        )


def write_sensitivity_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_fragment_charge_csv(
    path: Path,
    fragments: list[tuple[str, list[int]]],
    dft_fit: FitResult,
    mace_fit: FitResult,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fragment", "atom_indices_1based", "dft_charge_sum_e", "mace_charge_sum_e", "delta_e"],
        )
        writer.writeheader()
        for name, indices_1based in fragments:
            zero_based = np.array(indices_1based, dtype=int) - 1
            writer.writerow(
                {
                    "fragment": name,
                    "atom_indices_1based": ",".join(str(index) for index in indices_1based),
                    "dft_charge_sum_e": f"{float(np.sum(dft_fit.charges_e[zero_based])):.12g}",
                    "mace_charge_sum_e": f"{float(np.sum(mace_fit.charges_e[zero_based])):.12g}",
                    "delta_e": f"{float(np.sum(mace_fit.charges_e[zero_based] - dft_fit.charges_e[zero_based])):.12g}",
                }
            )


def write_diagnostics_text(
    path: Path,
    structure: Atoms,
    probe_set: ProbeSet,
    total_charge_e: int,
    spin_multiplicity: int,
    orca_cube_path: Path,
    generated_density_name: str | None,
    dft_fit: FitResult,
    mace_fit: FitResult,
    orca_potential_unit: str,
    pyresp_exe: str,
    resp_qwt: float,
) -> None:
    lines = [
        f"orca_cube_path={orca_cube_path}",
        f"generated_density_name={generated_density_name or ''}",
        f"natoms={len(structure)}",
        f"charge_e={total_charge_e}",
        f"spin_multiplicity={spin_multiplicity}",
        f"probe_distance_cutoff_A={probe_set.selected_cutoff_A:.12g}",
        f"available_probe_points={probe_set.available_count}",
        f"selected_probe_points={probe_set.selected_count}",
        f"probe_points_downsampled={str(probe_set.sampled).lower()}",
        f"min_selected_probe_distance_A={float(np.min(probe_set.min_atom_distance_A)):.12g}",
        f"max_selected_probe_distance_A={float(np.max(probe_set.min_atom_distance_A)):.12g}",
        f"orca_potential_unit_input={orca_potential_unit}",
        f"orca_potential_conversion_to_volt={'27.211386245988' if orca_potential_unit == 'au' else '1.0'}",
        f"fit_backend=py_resp.py",
        f"pyresp_exe={pyresp_exe}",
        f"resp_qwt={resp_qwt:.12g}",
        f"probe_identity_check=same_points_used_for_dft_and_mace_by_construction",
        f"dft_charge_sum_e={float(np.sum(dft_fit.charges_e)):.12g}",
        f"mace_charge_sum_e={float(np.sum(mace_fit.charges_e)):.12g}",
        f"dft_total_charge_constraint_error_e={float(np.sum(dft_fit.charges_e) - total_charge_e):.12g}",
        f"mace_total_charge_constraint_error_e={float(np.sum(mace_fit.charges_e) - total_charge_e):.12g}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_fragments(fragments: list[tuple[str, list[int]]], natoms: int) -> None:
    for name, indices in fragments:
        if max(indices) > natoms:
            raise ValueError(
                f"Fragment {name!r} references atom {max(indices)}, but the structure has only {natoms} atoms"
            )


def main() -> None:
    args = parse_args()
    if len(args.external_field) != 3:
        raise ValueError("--external-field must contain exactly three components")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    orca_cube_path, generated_density_name = ensure_orca_cube(args)
    cube_atoms, cube_data, origin_A, spacing_A = read_orca_cube(orca_cube_path)
    structure = load_structure(args, cube_atoms)
    total_charge_e, spin_multiplicity = resolve_charge_and_spin(args, structure)
    validate_fragments(args.fragment, len(structure))

    probe_set = build_probe_set(
        cube_data=cube_data,
        origin_A=origin_A,
        spacing_A=spacing_A,
        atom_positions_A=structure.positions,
        orca_potential_unit=args.orca_potential_unit,
        min_probe_distance_A=args.min_probe_distance_A,
        max_probes=args.max_probes,
    )

    status(
        f"Loaded {len(structure)} atoms and selected {probe_set.selected_count} probe points "
        f"with min atom-probe distance >= {probe_set.selected_cutoff_A:.3f} A"
    )
    calculator = build_mace_calculator(
        model=args.mace_model,
        device=args.device,
        dtype=args.dtype,
    )
    mace_atoms = configure_mace_atoms(
        structure,
        total_charge_e=total_charge_e,
        spin_multiplicity=spin_multiplicity,
        external_field=args.external_field,
    )
    density_coefficients = density_coefficients_from_calculator(calculator, mace_atoms)
    density_smearing_width_A, displaced_charge_offset_A = mace_descriptor_parameters(calculator)
    mace_esp_volt = evaluate_mace_esp_volt(
        probe_points_A=probe_set.points_A,
        density_coefficients=density_coefficients,
        atom_positions_A=structure.positions,
        density_smearing_width_A=density_smearing_width_A,
        displaced_charge_offset_A=displaced_charge_offset_A,
    )

    pyresp_root_dir = args.output_dir / "pyresp_runs"
    dft_fit = fit_with_pyresp(
        pyresp_exe=args.pyresp_exe,
        run_dir=pyresp_root_dir / "main_dft",
        title="DFT RESP fit",
        atoms=structure,
        probe_points_A=probe_set.points_A,
        target_volt=probe_set.values_volt,
        total_charge_e=total_charge_e,
        qwt=args.resp_qwt,
        free_hydrogens=args.resp_free_hydrogens,
    )
    mace_fit = fit_with_pyresp(
        pyresp_exe=args.pyresp_exe,
        run_dir=pyresp_root_dir / "main_mace",
        title="MACE RESP fit",
        atoms=structure,
        probe_points_A=probe_set.points_A,
        target_volt=mace_esp_volt,
        total_charge_e=total_charge_e,
        qwt=args.resp_qwt,
        free_hydrogens=args.resp_free_hydrogens,
    )

    sensitivity = sensitivity_rows(
        cutoffs_A=args.sensitivity_cutoffs_A,
        cube_data=cube_data,
        origin_A=origin_A,
        spacing_A=spacing_A,
        atoms=structure,
        dft_potential_unit=args.orca_potential_unit,
        max_probes=args.max_probes,
        mace_esp_full_volt=None,
        density_coefficients=density_coefficients,
        density_smearing_width_A=density_smearing_width_A,
        displaced_charge_offset_A=displaced_charge_offset_A,
        total_charge_e=total_charge_e,
        pyresp_exe=args.pyresp_exe,
        pyresp_root_dir=pyresp_root_dir / "sensitivity",
        resp_qwt=args.resp_qwt,
        resp_free_hydrogens=args.resp_free_hydrogens,
    )

    atomic_charges_csv = args.output_dir / "atomic_charges.csv"
    metrics_csv = args.output_dir / "esp_fit_metrics.csv"
    sensitivity_csv = args.output_dir / "probe_distance_sensitivity.csv"
    diagnostics_txt = args.output_dir / "diagnostics.txt"
    fragment_csv = args.output_dir / "fragment_charge_sums.csv"

    write_atomic_charges_csv(atomic_charges_csv, structure, dft_fit, mace_fit)
    write_metrics_csv(
        metrics_csv,
        dft_fit=dft_fit,
        mace_fit=mace_fit,
        probe_set=probe_set,
        total_charge_e=total_charge_e,
        spin_multiplicity=spin_multiplicity,
        density_smearing_width_A=density_smearing_width_A,
        displaced_charge_offset_A=displaced_charge_offset_A,
        orca_potential_unit=args.orca_potential_unit,
        resp_qwt=args.resp_qwt,
    )
    write_sensitivity_csv(sensitivity_csv, sensitivity)
    write_diagnostics_text(
        diagnostics_txt,
        structure=structure,
        probe_set=probe_set,
        total_charge_e=total_charge_e,
        spin_multiplicity=spin_multiplicity,
        orca_cube_path=orca_cube_path,
        generated_density_name=generated_density_name,
        dft_fit=dft_fit,
        mace_fit=mace_fit,
        orca_potential_unit=args.orca_potential_unit,
        pyresp_exe=args.pyresp_exe,
        resp_qwt=args.resp_qwt,
    )
    if args.fragment:
        write_fragment_charge_csv(fragment_csv, args.fragment, dft_fit, mace_fit)

    status(f"Wrote {atomic_charges_csv}")
    status(f"Wrote {metrics_csv}")
    status(f"Wrote {sensitivity_csv}")
    if args.fragment:
        status(f"Wrote {fragment_csv}")
    status(
        "DFT fit RMSE = "
        f"{dft_fit.rmse_volt:.6f} V; "
        "MACE fit RMSE = "
        f"{mace_fit.rmse_volt:.6f} V"
    )


if __name__ == "__main__":
    main()
