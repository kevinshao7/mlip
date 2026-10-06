#!/usr/bin/env python3
"""Keep every 100th conditionsfine XYZ frame and record its physical time."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MLIP_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MLIP_ROOT = Path(os.environ.get("MLIP_ROOT", LOCAL_MLIP_ROOT))
DEFAULT_INPUT_ROOT = DEFAULT_MLIP_ROOT / "outputsfull" / "conditionsfine"
DEFAULT_OUTPUT_ROOT = DEFAULT_MLIP_ROOT / "outputsfull" / "conditionsfinereduced"
DEFAULT_STRIDE = 100
# B2 production uses saveinterval=5 and MDtimestep=0.5 fs.
DEFAULT_SOURCE_INTERVAL_FS = 2.5
DEFAULT_FIRST_FRAME_TIME_FS = 2.5


@dataclass(frozen=True)
class Task:
    run_id: str
    source: Path
    destination: Path
    stride: int
    source_interval_fs: float
    first_frame_time_fs: float
    force: bool


@dataclass(frozen=True)
class Result:
    run_id: str
    status: str
    total_frames: int
    kept_frames: int
    stride: int
    source_frame_interval_fs: float
    reduced_frame_interval_fs: float
    first_kept_time_fs: float
    last_kept_time_fs: float | str
    input_bytes: int
    output_bytes: int
    input_xyz: str
    output_xyz: str
    message: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stride", type=positive_int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--source-frame-interval-fs", type=positive_float, default=DEFAULT_SOURCE_INTERVAL_FS,
        help="Time between input frames (default: 2.5 fs = 5 steps x 0.5 fs/step).",
    )
    parser.add_argument(
        "--first-frame-time-fs", type=float, default=DEFAULT_FIRST_FRAME_TIME_FS,
        help="Physical time assigned to input frame zero (default: 2.5 fs).",
    )
    parser.add_argument("--workers", type=positive_int,
                        default=max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    parser.add_argument("--run-id", action="append", help="Process only this run directory; repeatable.")
    parser.add_argument("--force", action="store_true", help="Replace existing reduced trajectories.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_tasks(args: argparse.Namespace) -> list[Task]:
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    requested = set(args.run_id or [])
    tasks: list[Task] = []
    for run_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        if requested and run_dir.name not in requested:
            continue
        trajectories = sorted(run_dir.glob("production*.xyz"))
        if len(trajectories) != 1:
            raise RuntimeError(
                f"Expected exactly one production*.xyz in {run_dir}, found {len(trajectories)}"
            )
        source = trajectories[0]
        destination = args.output_root / run_dir.name / source.name
        tasks.append(Task(run_dir.name, source, destination, args.stride,
                          args.source_frame_interval_fs, args.first_frame_time_fs, args.force))
    missing = requested - {task.run_id for task in tasks}
    if missing:
        raise FileNotFoundError(f"Requested run directories not found: {', '.join(sorted(missing))}")
    if not tasks:
        raise FileNotFoundError(f"No production trajectories found under {args.input_root}")
    return tasks


def parse_natoms(line: bytes, path: Path, frame_index: int) -> int:
    try:
        natoms = int(line.strip())
    except ValueError as exc:
        raise ValueError(f"{path}: invalid atom count at frame {frame_index}: {line!r}") from exc
    if natoms < 1:
        raise ValueError(f"{path}: non-positive atom count at frame {frame_index}: {natoms}")
    return natoms


def annotated_comment(original: bytes, task: Task, frame_index: int) -> bytes:
    newline = b"\r\n" if original.endswith(b"\r\n") else b"\n"
    content = original.rstrip(b"\r\n")
    source_time_fs = task.first_frame_time_fs + frame_index * task.source_interval_fs
    fields = (
        f" source_frame_index={frame_index}"
        f" source_time_fs={source_time_fs:.12g}"
        f" source_frame_interval_fs={task.source_interval_fs:.12g}"
        f" reduction_stride={task.stride}"
        f" reduced_frame_interval_fs={task.stride * task.source_interval_fs:.12g}"
    ).encode("ascii")
    return content + fields + newline


def reduce_one(task: Task) -> Result:
    total = kept = 0
    last_time: float | str = ""
    tmp = task.destination.with_suffix(task.destination.suffix + ".tmp")
    try:
        if task.destination.exists() and not task.force:
            raise FileExistsError(f"Output exists (pass --force to replace): {task.destination}")
        task.destination.parent.mkdir(parents=True, exist_ok=True)
        with task.source.open("rb") as source, tmp.open("wb") as destination:
            while True:
                natoms_line = source.readline()
                if natoms_line == b"":
                    break
                frame_index = total
                natoms = parse_natoms(natoms_line, task.source, frame_index)
                comment = source.readline()
                if comment == b"":
                    raise EOFError(f"{task.source}: missing comment at frame {frame_index}")
                keep = frame_index % task.stride == 0
                if keep:
                    destination.write(natoms_line)
                    destination.write(annotated_comment(comment, task, frame_index))
                for atom_index in range(natoms):
                    atom_line = source.readline()
                    if atom_line == b"":
                        raise EOFError(
                            f"{task.source}: unexpected EOF at frame {frame_index}, atom {atom_index}"
                        )
                    if keep:
                        destination.write(atom_line)
                total += 1
                if keep:
                    kept += 1
                    last_time = task.first_frame_time_fs + frame_index * task.source_interval_fs
        tmp.replace(task.destination)
        return Result(task.run_id, "ok", total, kept, task.stride, task.source_interval_fs,
                      task.stride * task.source_interval_fs, task.first_frame_time_fs, last_time,
                      task.source.stat().st_size, task.destination.stat().st_size,
                      str(task.source), str(task.destination), "")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return Result(task.run_id, "error", total, kept, task.stride, task.source_interval_fs,
                      task.stride * task.source_interval_fs, task.first_frame_time_fs, last_time,
                      task.source.stat().st_size if task.source.exists() else 0, 0,
                      str(task.source), str(task.destination), str(exc))


def run_tasks(tasks: list[Task], workers: int) -> list[Result]:
    if workers == 1:
        return [reduce_one(task) for task in tasks]
    results: list[Result] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(reduce_one, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            print(f"{result.run_id}: {result.status}, kept {result.kept_frames}/{result.total_frames}", flush=True)
            results.append(result)
    return results


def write_manifest(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in sorted(results, key=lambda item: item.run_id)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    tasks = discover_tasks(args)
    print(f"Found {len(tasks)} trajectories; keeping every {args.stride}th frame")
    print(f"Input spacing: {args.source_frame_interval_fs:g} fs; output spacing: "
          f"{args.stride * args.source_frame_interval_fs:g} fs")
    if args.dry_run:
        for task in tasks:
            print(f"DRY RUN {task.source} -> {task.destination}")
        return 0
    results = run_tasks(tasks, args.workers)
    manifest = args.output_root / "reduction_manifest.csv"
    write_manifest(manifest, results)
    failed = [result for result in results if result.status != "ok"]
    print(f"Manifest: {manifest}")
    if failed:
        for result in failed:
            print(f"ERROR {result.run_id}: {result.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
