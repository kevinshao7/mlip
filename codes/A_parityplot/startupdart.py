#!/usr/bin/env python3
"""Generate direct-run ORCA jobs for DART/stormy parity-plot clusters.

These machines are not Slurm/sbatch machines. This launcher writes ORCA input
files using the same FairChem/ASE path as startup.py, then writes plain shell
runners that run ORCA jobs in series. The generated shell runners do not need
Python, ASE, or FairChem on the HPC machine.

Examples:
    python startupdart.py --prepare-all --frames 0,180
    python startupdart.py --machine stormy --frames 0,45 --generate-only
    python startupdart.py --machine dart9 --frames 0,25 --resume
    python startupdart.py --machine dart10 --frames 25,50
    python startupdart.py --machine dart11 --frames 50,75 --module mpi/openmpi-x86_64

Prepare a four-machine split for frames 0-180:
    python startupdart.py --prepare-all --frames 0,180 --force

Then run one shell command on each machine. Each command runs its assigned ORCA
jobs in series, with no Python command involved:
    cd codes/A_parityplot/8_4_stormy && bash run_stormy.sh
    cd codes/A_parityplot/8_5_dart9 && bash run_dart9.sh
    cd codes/A_parityplot/8_6_dart10 && bash run_dart10.sh
    cd codes/A_parityplot/8_7_dart11 && bash run_dart11.sh

If all four folders are on one host and you intentionally want one serial run:
    cd codes/A_parityplot && bash run_all_dart_serial.sh

The frame range is half-open: --frames 0,25 means cluster frames 0 through 24.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import startup


SCRIPT_DIR = Path(__file__).resolve().parent
AP_DIR = SCRIPT_DIR
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CLUSTER_XYZ = startup.DEFAULT_CLUSTER_XYZ
DEFAULT_ORCA_COMMAND = "orca_qc"
DEFAULT_THREADS = 12
DEFAULT_HPC_MLIP_DIR = startup.DEFAULT_HPC_MLIP_DIR


MACHINES = {
    "stormy": startup.MachineConfig(
        name="stormy",
        job_dir=AP_DIR / "8_4_stormy",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_4_stormy",
        stem_prefix="r09_hot_w_isolatedH_stormy",
    ),
    "dart9": startup.MachineConfig(
        name="dart9",
        job_dir=AP_DIR / "8_5_dart9",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_5_dart9",
        stem_prefix="r09_hot_w_isolatedH_dart9",
    ),
    "dart10": startup.MachineConfig(
        name="dart10",
        job_dir=AP_DIR / "8_6_dart10",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_6_dart10",
        stem_prefix="r09_hot_w_isolatedH_dart10",
    ),
    "dart11": startup.MachineConfig(
        name="dart11",
        job_dir=AP_DIR / "8_7_dart11",
        output_dir=MLIP_DIR / "outputsfull" / "A_parityplot" / "8_7_dart11",
        stem_prefix="r09_hot_w_isolatedH_dart11",
    ),
}

PREPARE_ALL_MACHINE_ORDER = ["stormy", "dart9", "dart10", "dart11"]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def parse_task_indices(spec: str | None, start: int, stop: int) -> list[int] | None:
    if not spec:
        return None

    indices: list[int] = []
    for field in spec.split(","):
        field = field.strip()
        if not field:
            continue
        if "-" in field:
            left, right = field.split("-", 1)
            first = int(left)
            last = int(right)
            if last < first:
                fail(f"--task-indices range must be ascending, got {field!r}")
            indices.extend(range(first, last + 1))
        else:
            indices.append(int(field))

    bad = [idx for idx in indices if idx < start or idx >= stop]
    if bad:
        fail(f"--task-indices contains frames outside {start},{stop}: {bad}")
    return sorted(dict.fromkeys(indices))


def split_frame_range(start: int, stop: int, names: list[str]) -> dict[str, tuple[int, int]]:
    total = stop - start
    base = total // len(names)
    remainder = total % len(names)
    ranges: dict[str, tuple[int, int]] = {}
    cursor = start
    for index, name in enumerate(names):
        count = base + (1 if index < remainder else 0)
        ranges[name] = (cursor, cursor + count)
        cursor += count
    return ranges


def command_for_orca(inp_path: Path, orca_command: str, modules: list[str], pre_command: str | None) -> list[str]:
    command = " ".join([shlex.quote(part) for part in shlex.split(orca_command)])
    command = f"{command} {shlex.quote(str(inp_path))}"

    prefix: list[str] = []
    for module in modules:
        module = module.strip()
        if module:
            prefix.append(f"module load {shlex.quote(module)}")
    if pre_command:
        prefix.append(pre_command)

    if os.name == "posix" and prefix:
        return ["bash", "-lc", " && ".join(prefix + [f"exec {command}"])]
    return shlex.split(orca_command) + [str(inp_path)]


def runner_path(config: startup.MachineConfig) -> Path:
    return config.job_dir / f"run_{config.name}.sh"


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def runner_text(
    config: startup.MachineConfig,
    frame_indices: list[int],
    orca_command: str,
    modules: list[str],
    pre_command: str | None,
    threads: int,
    hpc_mlip_dir: str,
) -> str:
    job_dir_rel = startup.relative_to_mlip(config.job_dir)
    output_dir_rel = startup.relative_to_mlip(config.output_dir)
    stems = [startup.stem_for_frame(config, frame_index) for frame_index in frame_indices]
    stem_lines = "\n".join(f"    {shell_quote(stem)}" for stem in stems)

    module_lines = []
    for module in modules:
        module = module.strip()
        if module:
            module_lines.append(
                f"type module >/dev/null 2>&1 && module load {shell_quote(module)}"
            )
    if pre_command:
        module_lines.append(pre_command)
    setup_lines = "\n".join(module_lines)
    if setup_lines:
        setup_lines += "\n"

    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MLIP_DIR="${{MLIP_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"
INPUT_DIR="$MLIP_DIR/{job_dir_rel}"
OUTPUT_DIR="$MLIP_DIR/{output_dir_rel}"
ORCA_COMMAND={shell_quote(orca_command)}
BASIS_FILE="{startup.BASIS_FILE}"
FORCE="${{FORCE:-1}}"

export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
export OPENBLAS_NUM_THREADS={threads}

{setup_lines}mkdir -p "$OUTPUT_DIR"
if [[ -f "$INPUT_DIR/$BASIS_FILE" ]]; then
    cp "$INPUT_DIR/$BASIS_FILE" "$OUTPUT_DIR/$BASIS_FILE"
else
    echo "Missing ORCA basis file: $INPUT_DIR/$BASIS_FILE" >&2
    exit 1
fi
cd "$OUTPUT_DIR"

STEMS=(
{stem_lines}
)

for STEM in "${{STEMS[@]}}"; do
    INPUT_PATH="$INPUT_DIR/$STEM.inp"
    OUTPUT_PATH="$OUTPUT_DIR/$STEM.out"

    if [[ ! -f "$INPUT_PATH" ]]; then
        echo "Missing ORCA input: $INPUT_PATH" >&2
        exit 1
    fi

    if [[ -f "$OUTPUT_PATH" ]] && grep -q "{startup.FINAL_ENERGY_MARKER}" "$OUTPUT_PATH" && grep -q "{startup.NORMAL_TERMINATION_MARKER}" "$OUTPUT_PATH"; then
        echo "Skipping completed $OUTPUT_PATH"
        continue
    fi

    if [[ -f "$OUTPUT_PATH" ]]; then
        if [[ "$FORCE" == "1" ]]; then
            rm -f "$OUTPUT_PATH"
        else
            echo "Output already exists or is incomplete: $OUTPUT_PATH" >&2
            echo "Use FORCE=1 bash $(basename "$0") to overwrite incomplete output." >&2
            exit 1
        fi
    fi

    echo "Running $INPUT_PATH -> $OUTPUT_PATH"
    $ORCA_COMMAND "$INPUT_PATH" > "$OUTPUT_PATH"

    grep -q "{startup.FINAL_ENERGY_MARKER}" "$OUTPUT_PATH"
    grep -q "{startup.NORMAL_TERMINATION_MARKER}" "$OUTPUT_PATH"
done
"""


def write_runner(
    config: startup.MachineConfig,
    frame_indices: list[int],
    orca_command: str,
    modules: list[str],
    pre_command: str | None,
    threads: int,
    hpc_mlip_dir: str,
) -> Path:
    path = runner_path(config)
    startup.write_text_lf(
        path,
        runner_text(config, frame_indices, orca_command, modules, pre_command, threads, hpc_mlip_dir),
    )
    return path


def write_master_runner(machine_names: list[str], hpc_mlip_dir: str) -> Path:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'MLIP_DIR="${MLIP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"',
        "",
    ]
    for name in machine_names:
        config = MACHINES[name]
        job_dir_rel = startup.relative_to_mlip(config.job_dir)
        lines.append(f'bash "$MLIP_DIR/{job_dir_rel}/{runner_path(config).name}"')
    path = AP_DIR / "run_all_dart_serial.sh"
    startup.write_text_lf(path, "\n".join(lines) + "\n")
    return path


def direct_orca_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    return env


def run_orca_direct(
    config: startup.MachineConfig,
    inp_path: Path,
    orca_command: str,
    modules: list[str],
    pre_command: str | None,
    threads: int,
    resume: bool,
    force: bool,
) -> None:
    out_path = config.output_dir / f"{inp_path.stem}.out"
    if out_path.exists():
        has_energy, terminated = startup.parse_output_status(out_path)
        if resume and has_energy and terminated:
            print(f"Skipping completed {out_path.name}", flush=True)
            return
        if force:
            out_path.unlink()
        else:
            fail(f"Output already exists or is incomplete: {out_path}. Use --resume or --force.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = direct_orca_env(threads)

    print(f"Running {inp_path.name} -> {out_path}", flush=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        process = subprocess.Popen(
            command_for_orca(inp_path, orca_command, modules, pre_command),
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
    has_energy, terminated = startup.parse_output_status(out_path)
    if not has_energy or not terminated:
        fail(f"ORCA output did not complete normally: {out_path}")


def selected_jobs(
    frames: list,
    start: int,
    stop: int,
    task_indices: list[int] | None,
) -> list[tuple[int, object]]:
    if task_indices is None:
        return [(frame_index, atoms) for frame_index, atoms in zip(range(start, stop), frames)]
    by_index = {frame_index: atoms for frame_index, atoms in zip(range(start, stop), frames)}
    return [(frame_index, by_index[frame_index]) for frame_index in task_indices]


def prepare_machine(
    config: startup.MachineConfig,
    frames_spec: str,
    clusters: Path,
    task_indices_spec: str | None,
    orca_command: str,
    modules: list[str],
    pre_command: str | None,
    threads: int,
    hpc_mlip_dir: str,
    force: bool,
    dry_run: bool,
) -> tuple[list[Path], Path | None, list[tuple[int, object]]]:
    start, stop = startup.parse_frames(frames_spec)
    task_indices = parse_task_indices(task_indices_spec, start, stop)
    frames = startup.load_cluster_frames(clusters, start, stop)
    jobs = selected_jobs(frames, start, stop, task_indices)

    print(f"Machine: {config.name}")
    print(f"Cluster XYZ: {clusters}")
    print(f"Frame range: {start},{stop}")
    print(f"Selected jobs: {len(jobs)}")
    print(f"Job directory: {config.job_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"ORCA command: {orca_command}")
    if modules:
        print(f"Modules: {', '.join(modules)}")

    if dry_run:
        for frame_index, _atoms in jobs:
            stem = startup.stem_for_frame(config, frame_index)
            print(f"{frame_index:03d} {config.job_dir / (stem + '.inp')} -> {config.output_dir / (stem + '.out')}")
        return [], None, jobs

    startup.THREADS = threads
    generated: list[Path] = []
    for frame_index, atoms in jobs:
        inp_path = startup.generate_input(config, atoms, frame_index, force=force)
        generated.append(inp_path)
        print(f"Prepared {inp_path}")

    frame_indices = [frame_index for frame_index, _atoms in jobs]
    script_path = write_runner(
        config,
        frame_indices,
        orca_command,
        modules,
        pre_command,
        threads,
        hpc_mlip_dir,
    )
    print(f"Prepared {script_path}")
    return generated, script_path, jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", choices=sorted(MACHINES))
    parser.add_argument(
        "--prepare-all",
        action="store_true",
        help="Prepare an even four-way split for stormy, dart9, dart10, and dart11.",
    )
    parser.add_argument("--frames", required=True, help="Half-open cluster frame range, e.g. 0,25")
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTER_XYZ)
    parser.add_argument(
        "--task-indices",
        default=None,
        help="Optional absolute frame indices to run inside --frames, e.g. 0,3,7-10.",
    )
    parser.add_argument("--orca-command", default=DEFAULT_ORCA_COMMAND)
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Optional environment module to load before each ORCA run. Repeat for multiple modules.",
    )
    parser.add_argument(
        "--pre-command",
        default=None,
        help="Optional shell command to run before ORCA, after module loads.",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--hpc-mlip-dir",
        default=DEFAULT_HPC_MLIP_DIR,
        help=f"MLIP root path used inside generated shell runners. Default: {DEFAULT_HPC_MLIP_DIR}",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.threads < 1:
        fail("--threads must be at least 1")

    if args.prepare_all and args.machine:
        fail("Use either --prepare-all or --machine, not both")
    if args.prepare_all and args.task_indices:
        fail("--task-indices is only supported with --machine")
    if not args.prepare_all and not args.machine:
        fail("Pass --machine or --prepare-all")

    if args.prepare_all:
        startup.load_fairchem_orca_calc()
        start, stop = startup.parse_frames(args.frames)
        ranges = split_frame_range(start, stop, PREPARE_ALL_MACHINE_ORDER)
        total_inputs = 0
        for name in PREPARE_ALL_MACHINE_ORDER:
            split_start, split_stop = ranges[name]
            print()
            generated, script_path, _jobs = prepare_machine(
                MACHINES[name],
                f"{split_start},{split_stop}",
                args.clusters,
                None,
                args.orca_command,
                args.module,
                args.pre_command,
                args.threads,
                args.hpc_mlip_dir,
                args.force,
                args.dry_run,
            )
            total_inputs += len(generated)
            if script_path:
                print(f"Runner: cd {startup.relative_to_mlip(MACHINES[name].job_dir)} && bash {script_path.name}")

        if args.dry_run:
            return
        master = write_master_runner(PREPARE_ALL_MACHINE_ORDER, args.hpc_mlip_dir)
        print()
        print(f"Generated {total_inputs} input files")
        print(f"Prepared {master}")
        print(f"Single-host serial runner: cd {startup.relative_to_mlip(AP_DIR)} && bash {master.name}")
        return

    startup.load_fairchem_orca_calc()

    config = MACHINES[args.machine]
    generated, script_path, _jobs = prepare_machine(
        config,
        args.frames,
        args.clusters,
        args.task_indices,
        args.orca_command,
        args.module,
        args.pre_command,
        args.threads,
        args.hpc_mlip_dir,
        args.force,
        args.dry_run,
    )

    if args.dry_run:
        return

    if args.generate_only:
        print(f"Generated {len(generated)} input files")
        if script_path:
            print(f"Runner: cd {startup.relative_to_mlip(config.job_dir)} && bash {script_path.name}")
        return

    for inp_path in generated:
        run_orca_direct(
            config,
            inp_path,
            args.orca_command,
            args.module,
            args.pre_command,
            args.threads,
            resume=args.resume,
            force=args.force,
        )


if __name__ == "__main__":
    main()
