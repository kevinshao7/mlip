#!/usr/bin/env python3
"""Generate and run ORCA jobs for isolated-H validation clusters.

Examples:
    python startup.py --machine bluehive --frames 0,180 --force

for f in 8_5_bluehiveDFT/r09_hot_w_isolatedH_bluehive_*.slurm; do sbatch "$f"; done

The frame range is half-open: --frames 0,100 means cluster frames 0 through 99.
Running without --task-index writes one .inp and one .slurm file per selected frame.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import read


SCRIPT_DIR = Path(__file__).resolve().parent
AP_DIR = SCRIPT_DIR
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CLUSTER_XYZ = AP_DIR / "7_26_H2pathvalidation" / "r09_hot_w_h2formation_training_clusters.xyz"
FAIRCHEM_ORCA_CALC = MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "calc.py"
FAIRCHEM_SRC = MLIP_DIR / "fairchem" / "src"
FAIRCHEM_ORCA_BASIS = (
    MLIP_DIR / "fairchem" / "src" / "fairchem" / "data" / "omol" / "orca" / "basis" / "def2-tzvpd.bas"
)
BASIS_FILE = "def2-tzvpd.bas"
DEFAULT_ORCA_COMMAND = "orca"
THREADS = 24
MULTIPLICITY = 1
DEFAULT_HPC_MLIP_DIR = "/home/kevinsh/kevinsh/mlip"
DEFAULT_ORCA_MODULE = "orca/6.0.1"
FINAL_ENERGY_MARKER = "FINAL SINGLE POINT ENERGY"
NORMAL_TERMINATION_MARKER = "ORCA TERMINATED NORMALLY"

ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
}
FORMAL_CHARGES = {
    "H": 1,
    "N": -3,
    "O": -2,
    "S": -2,
}


@dataclass(frozen=True)
class MachineConfig:
    name: str
    job_dir: Path
    output_dir: Path
    stem_prefix: str
    slurm_time: str = "6:00:00"
    slurm_memory: str | None = "64G"
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_mail_type: str | None = None
    slurm_mail_user: str | None = None
    slurm_exclude: str | None = None
    slurm_output: str | None = None
    slurm_error: str | None = None
    slurm_use_cpus_per_task: bool = True
    threads: int = THREADS
    orca_module: str = DEFAULT_ORCA_MODULE
    hpc_mlip_dir: str = DEFAULT_HPC_MLIP_DIR


MACHINES = {
    "viper": MachineConfig(
        name="viper",
        job_dir=AP_DIR / "8_1_viperDFT",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_1_viperDFT",
        stem_prefix="r09_hot_w_isolatedH_viper",
    ),
    "raven": MachineConfig(
        name="raven",
        job_dir=AP_DIR / "8_2_ravenDFT",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_2_ravenDFT",
        stem_prefix="r09_hot_w_isolatedH_raven",
    ),
    "greatlakes": MachineConfig(
        name="greatlakes",
        job_dir=AP_DIR / "8_4_greatlakesDFT",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_4_greatlakesDFT",
        stem_prefix="r09_hot_w_isolatedH_greatlakes",
        slurm_time="6:00:00",
        slurm_memory="31G",
        slurm_partition="standard",
        slurm_account="chengcli0",
        threads=12,
        orca_module="orca/6.0.1",
        hpc_mlip_dir="/home/kevinsh/kevinsh/mlip",
    ),
    "bluehive": MachineConfig(
        name="bluehive",
        job_dir=AP_DIR / "8_5_bluehiveDFT",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_5_bluehiveDFT",
        stem_prefix="r09_hot_w_isolatedH_bluehive",
        slurm_time="12:00:00",
        slurm_partition="standard",
        slurm_mail_type="END",
        slurm_mail_user="ks2120@cam.ac.uk",
        slurm_exclude="bhc0020,bhc0021,bhc0025",
        slurm_output="{stem}.slurmlog.txt",
        slurm_error="{stem}.slurmerr.txt",
        slurm_use_cpus_per_task=False,
        slurm_memory=None,
        threads=24,
        orca_module="orca/6.1.1",
        hpc_mlip_dir="/gpfs/fs2/scratch/kshao4/mlip",
    ),
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def parse_frames(spec: str) -> tuple[int, int]:
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) != 2:
        fail(f"--frames must be start,stop, got {spec!r}")
    start, stop = (int(fields[0]), int(fields[1]))
    if start < 0 or stop <= start:
        fail(f"--frames must satisfy 0 <= start < stop, got {spec!r}")
    return start, stop


def atom_tuples(atoms: Atoms) -> list[tuple[str, float, float, float]]:
    rows = []
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"No atomic number configured for {symbol!r}")
        rows.append((symbol, float(position[0]), float(position[1]), float(position[2])))
    return rows


def formal_charge(atoms: list[tuple[str, float, float, float]]) -> int:
    charge = 0
    for symbol, *_ in atoms:
        try:
            charge += FORMAL_CHARGES[symbol]
        except KeyError as exc:
            raise ValueError(f"No formal charge configured for {symbol!r}") from exc
    return charge


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def load_fairchem_orca_calc():
    if not FAIRCHEM_SRC.is_dir():
        fail(f"Required FairChem source directory is missing: {FAIRCHEM_SRC}")
    if not FAIRCHEM_ORCA_CALC.is_file():
        fail(f"Required FairChem ORCA calc module is missing: {FAIRCHEM_ORCA_CALC}")
    if not FAIRCHEM_ORCA_BASIS.is_file():
        fail(f"Required FairChem ORCA basis file is missing: {FAIRCHEM_ORCA_BASIS}")

    sys.path.insert(0, str(FAIRCHEM_SRC))

    try:
        return importlib.import_module("fairchem.data.omol.orca.calc")
    except ImportError as import_error:
        module_import_error = import_error

    spec = importlib.util.spec_from_file_location("fairchem_omol_orca_calc", FAIRCHEM_ORCA_CALC)
    if spec is None or spec.loader is None:
        fail(f"Could not load FairChem ORCA calc module from {FAIRCHEM_ORCA_CALC}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as file_load_error:
        fail(
            "FairChem ORCA support is required and could not be loaded. "
            f"Package import error: {module_import_error!r}; direct file-load error: {file_load_error!r}"
        )
    return module


def make_input_with_fairchem(atoms: Atoms, work_dir: Path, threads: int) -> str:
    orca_calc = load_fairchem_orca_calc()
    charge = formal_charge(atom_tuples(atoms))
    from ase.calculators.orca import OrcaProfile

    def compatible_orca_profile(command):
        if isinstance(command, list):
            command = command[0] or "orca"
        return OrcaProfile(command)

    orca_calc.OrcaProfile = compatible_orca_profile
    transient_input = work_dir / "orca.inp"
    orcasimpleinput = " ".join(
        token
        for token in orca_calc.ORCA_ASE_SIMPLE_INPUT.split()
        if not (token.upper().startswith("PAL") and token[3:].isdigit())
    )
    orca_calc.write_orca_inputs(
        atoms,
        work_dir,
        charge=charge,
        mult=MULTIPLICITY,
        orcasimpleinput=orcasimpleinput,
    )
    text = transient_input.read_text(encoding="utf-8")
    transient_input.unlink()
    lines = text.splitlines()
    if lines and not any(line.strip().lower().startswith("%pal") for line in lines):
        lines.insert(1, f"%pal nprocs {threads} end")
        text = "\n".join(lines) + "\n"
    return text


def load_cluster_frames(path: Path, start: int, stop: int) -> list[Atoms]:
    if not path.is_file():
        fail(f"Cluster XYZ not found: {path}")
    frames = read(path, ":")
    if len(frames) < stop:
        fail(f"Requested frames {start},{stop}, but {path} contains only {len(frames)} frames")
    return frames[start:stop]


def stem_for_frame(config: MachineConfig, frame_index: int) -> str:
    return f"{config.stem_prefix}_{frame_index:03d}"


def stage_basis(config: MachineConfig) -> None:
    if not FAIRCHEM_ORCA_BASIS.is_file():
        fail(f"Required fairchem ORCA basis file not found: {FAIRCHEM_ORCA_BASIS}")
    shutil.copy2(FAIRCHEM_ORCA_BASIS, config.job_dir / BASIS_FILE)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FAIRCHEM_ORCA_BASIS, config.output_dir / BASIS_FILE)


def generate_input(config: MachineConfig, atoms: Atoms, frame_index: int, force: bool) -> Path:
    config.job_dir.mkdir(parents=True, exist_ok=True)
    stage_basis(config)
    stem = stem_for_frame(config, frame_index)
    inp_path = config.job_dir / f"{stem}.inp"
    if inp_path.exists() and not force:
        return inp_path
    write_text_lf(inp_path, make_input_with_fairchem(atoms, config.job_dir, config.threads))
    return inp_path


def relative_to_mlip(path: Path) -> str:
    try:
        return path.resolve().relative_to(MLIP_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def slurm_text(config: MachineConfig, frame_index: int, orca_command: str) -> str:
    stem = stem_for_frame(config, frame_index)
    job_dir_rel = relative_to_mlip(config.job_dir)
    output_dir_rel = relative_to_mlip(config.output_dir)
    slurm_out_dir = f"{config.hpc_mlip_dir}/outputsfull/slurm"
    partition_line = f"#SBATCH --partition={config.slurm_partition}\n" if config.slurm_partition else ""
    account_line = f"#SBATCH --account={config.slurm_account}\n" if config.slurm_account else ""
    mail_type_line = f"#SBATCH --mail-type={config.slurm_mail_type}\n" if config.slurm_mail_type else ""
    mail_user_line = f"#SBATCH --mail-user={config.slurm_mail_user}\n" if config.slurm_mail_user else ""
    exclude_line = f"#SBATCH --exclude={config.slurm_exclude}\n" if config.slurm_exclude else ""
    ntasks_line = f"#SBATCH -n {config.threads}\n" if not config.slurm_use_cpus_per_task else "#SBATCH -n 1\n"
    long_ntasks_line = "#SBATCH --ntasks=1\n" if config.slurm_use_cpus_per_task else ""
    cpus_line = f"#SBATCH --cpus-per-task={config.threads}\n" if config.slurm_use_cpus_per_task else ""
    mem_line = f"#SBATCH --mem={config.slurm_memory}\n" if config.slurm_memory else ""
    output_line = (
        f"#SBATCH -o {config.slurm_output.format(stem=stem)}\n"
        if config.slurm_output
        else f"#SBATCH --output={slurm_out_dir}/{stem}-%j.out\n"
    )
    error_line = (
        f"#SBATCH -e {config.slurm_error.format(stem=stem)}\n"
        if config.slurm_error
        else f"#SBATCH --error={slurm_out_dir}/{stem}-%j.err\n"
    )
    return f"""#!/bin/bash
#SBATCH --job-name={stem}
{partition_line}{account_line}{mail_type_line}{mail_user_line}{ntasks_line}{long_ntasks_line}{cpus_line}{mem_line}{exclude_line}\
#SBATCH --time={config.slurm_time}
{output_line.rstrip()}
{error_line.rstrip()}

set -euo pipefail

module purge
module load {config.orca_module}

MLIP_DIR="${{MLIP_DIR:-{config.hpc_mlip_dir}}}"
INPUT_PATH="$MLIP_DIR/{job_dir_rel}/{stem}.inp"
OUTPUT_DIR="$MLIP_DIR/{output_dir_rel}"
OUTPUT_PATH="$OUTPUT_DIR/{stem}.out"
ORCA_COMMAND={orca_command!r}
ORCA_COMMAND="$(command -v "$ORCA_COMMAND")"

echo "MLIP_DIR=$MLIP_DIR"
echo "ORCA_COMMAND=$ORCA_COMMAND"
echo "INPUT_PATH=$INPUT_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"

mkdir -p "$MLIP_DIR/outputsfull/slurm"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Missing ORCA input: $INPUT_PATH" >&2
    exit 1
fi

if [[ -f "$OUTPUT_PATH" ]] && grep -q "{FINAL_ENERGY_MARKER}" "$OUTPUT_PATH" && grep -q "{NORMAL_TERMINATION_MARKER}" "$OUTPUT_PATH"; then
    echo "Skipping completed $OUTPUT_PATH"
    exit 0
fi

export OMP_NUM_THREADS={config.threads}
export MKL_NUM_THREADS={config.threads}
export OPENBLAS_NUM_THREADS={config.threads}

cd "$OUTPUT_DIR"
$ORCA_COMMAND "$INPUT_PATH" > "$OUTPUT_PATH"

grep -q "{FINAL_ENERGY_MARKER}" "$OUTPUT_PATH"
grep -q "{NORMAL_TERMINATION_MARKER}" "$OUTPUT_PATH"
"""


def generate_slurm(config: MachineConfig, frame_index: int, orca_command: str, force: bool) -> Path:
    config.job_dir.mkdir(parents=True, exist_ok=True)
    slurm_path = config.job_dir / f"{stem_for_frame(config, frame_index)}.slurm"
    if slurm_path.exists() and not force:
        return slurm_path
    write_text_lf(slurm_path, slurm_text(config, frame_index, orca_command))
    return slurm_path


def parse_output_status(out_path: Path) -> tuple[bool, bool]:
    if not out_path.is_file():
        return False, False
    text = out_path.read_text(encoding="utf-8", errors="replace")
    return FINAL_ENERGY_MARKER in text, NORMAL_TERMINATION_MARKER in text


def run_orca(config: MachineConfig, inp_path: Path, orca_command: str, resume: bool, force: bool) -> None:
    out_path = config.output_dir / f"{inp_path.stem}.out"
    if out_path.exists():
        has_energy, terminated = parse_output_status(out_path)
        if resume and has_energy and terminated:
            print(f"Skipping completed {out_path.name}", flush=True)
            return
        if force:
            out_path.unlink()
        else:
            fail(f"Output already exists or is incomplete: {out_path}. Use --resume or --force.")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config.threads)
    env["MKL_NUM_THREADS"] = str(config.threads)
    env["OPENBLAS_NUM_THREADS"] = str(config.threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {inp_path.name} -> {out_path}", flush=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        process = subprocess.Popen(
            [orca_command, str(inp_path)],
            cwd=config.output_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            out_handle.write(line)

    return_code = process.wait()
    if return_code != 0:
        fail(f"ORCA failed for {inp_path} with exit code {return_code}; output: {out_path}")
    has_energy, terminated = parse_output_status(out_path)
    if not has_energy or not terminated:
        fail(f"ORCA output did not complete normally: {out_path}")


def slurm_task_index(explicit_task_index: int | None) -> int | None:
    if explicit_task_index is not None:
        return explicit_task_index
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(value) if value not in (None, "") else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", choices=sorted(MACHINES), required=True)
    parser.add_argument("--frames", required=True, help="Half-open cluster frame range, e.g. 0,100")
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTER_XYZ)
    parser.add_argument("--task-index", type=int, default=None, help="Absolute cluster frame index to run.")
    parser.add_argument("--orca-command", default=DEFAULT_ORCA_COMMAND)
    parser.add_argument("--generate-only", action="store_true", help="Write inputs for the frame range without running ORCA.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_fairchem_orca_calc()

    start, stop = parse_frames(args.frames)
    config = MACHINES[args.machine]
    task_index = slurm_task_index(args.task_index)

    if task_index is not None and not start <= task_index < stop:
        fail(f"Task index {task_index} is outside requested frame range {start},{stop}")

    frames = load_cluster_frames(args.clusters, start, stop)
    selected = (
        [(task_index, frames[task_index - start])]
        if task_index is not None
        else [(frame_index, atoms) for frame_index, atoms in zip(range(start, stop), frames)]
    )

    print(f"Machine: {config.name}")
    print(f"Cluster XYZ: {args.clusters}")
    print(f"Frame range: {start},{stop}")
    print(f"Job directory: {config.job_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Selected jobs: {len(selected)}")

    if args.dry_run:
        for frame_index, _atoms in selected:
            stem = stem_for_frame(config, frame_index)
            print(f"{frame_index:03d} {config.job_dir / (stem + '.inp')} {config.job_dir / (stem + '.slurm')}")
        return

    generated: list[Path] = []
    generated_slurms: list[Path] = []
    for frame_index, atoms in selected:
        inp_path = generate_input(config, atoms, frame_index, force=args.force)
        slurm_path = generate_slurm(config, frame_index, args.orca_command, force=args.force)
        generated.append(inp_path)
        generated_slurms.append(slurm_path)
        print(f"Prepared {inp_path}")
        print(f"Prepared {slurm_path}")

    if args.generate_only or task_index is None:
        print(f"Generated {len(generated)} input files and {len(generated_slurms)} Slurm files")
        return

    run_orca(config, generated[0], args.orca_command, resume=args.resume, force=args.force)


if __name__ == "__main__":
    main()
