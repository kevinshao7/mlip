#!/usr/bin/env python3
"""Stream-copy every Nth frame from condition-production XYZ trajectories.

This deliberately does not parse or rewrite atom records with ASE. Selected
frames are copied byte-for-byte from the input trajectory.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MLIP_ROOT = Path(os.environ.get("MLIP_ROOT", "/dais/fs/scratch/kshao/mlip"))
DEFAULT_INPUT_ROOT = DEFAULT_MLIP_ROOT / "outputsfull" / "conditionsproduction"
DEFAULT_OUTPUT_ROOT = DEFAULT_MLIP_ROOT / "outputsfull" / "B1_conditionsproduction_stride100_xyz"
DEFAULT_STRIDE = 100


@dataclass(frozen=True)
class CondenseTask:
    run_id: str
    input_xyz: Path
    output_xyz: Path
    stride: int


@dataclass(frozen=True)
class CondenseResult:
    run_id: str
    input_xyz: str
    output_xyz: str
    stride: int
    input_bytes: int
    output_bytes: int
    total_frames: int
    kept_frames: int
    status: str
    message: str


def status(message: str) -> None:
    print(message, flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy every Nth frame from each .xyz trajectory under conditionsproduction. "
            "Frame contents are copied exactly; no chemistry preprocessing is performed."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stride", type=positive_int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))),
        help="Number of trajectories to stream in parallel.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Optional run directory name to process. May be repeated. Default: all run directories.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_tasks(input_root: Path, output_root: Path, stride: int, run_ids: list[str] | None) -> list[CondenseTask]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    selected = set(run_ids or [])
    tasks: list[CondenseTask] = []
    for run_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        run_id = run_dir.name
        if selected and run_id not in selected:
            continue
        xyz_files = sorted(run_dir.glob("*.xyz"))
        if len(xyz_files) != 1:
            raise RuntimeError(f"Expected exactly one .xyz in {run_dir}, found {len(xyz_files)}")
        input_xyz = xyz_files[0]
        output_xyz = output_root / run_id / f"{input_xyz.stem}_stride{stride}.xyz"
        tasks.append(CondenseTask(run_id, input_xyz, output_xyz, stride))

    missing = sorted(selected - {task.run_id for task in tasks})
    if missing:
        raise RuntimeError(f"Requested run-id(s) not found under {input_root}: {', '.join(missing)}")
    if not tasks:
        raise RuntimeError(f"No input trajectories found under {input_root}")
    return tasks


def parse_natoms(line: bytes, path: Path, frame_index: int) -> int:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"{path}: blank atom-count line at frame {frame_index}")
    try:
        natoms = int(stripped)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid atom-count line at frame {frame_index}: {line!r}") from exc
    if natoms < 0:
        raise ValueError(f"{path}: negative atom count at frame {frame_index}: {natoms}")
    return natoms


def condense_one(task: CondenseTask) -> CondenseResult:
    task.output_xyz.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = task.output_xyz.with_suffix(task.output_xyz.suffix + ".tmp")
    total_frames = 0
    kept_frames = 0

    try:
        with task.input_xyz.open("rb") as src, tmp_output.open("wb") as dst:
            while True:
                natoms_line = src.readline()
                if natoms_line == b"":
                    break

                frame_index = total_frames
                natoms = parse_natoms(natoms_line, task.input_xyz, frame_index)
                comment_line = src.readline()
                if comment_line == b"":
                    raise EOFError(f"{task.input_xyz}: missing comment line at frame {frame_index}")

                keep = frame_index % task.stride == 0
                if keep:
                    dst.write(natoms_line)
                    dst.write(comment_line)

                for atom_index in range(natoms):
                    atom_line = src.readline()
                    if atom_line == b"":
                        raise EOFError(
                            f"{task.input_xyz}: unexpected EOF at frame {frame_index}, atom {atom_index}"
                        )
                    if keep:
                        dst.write(atom_line)

                total_frames += 1
                if keep:
                    kept_frames += 1

        tmp_output.replace(task.output_xyz)
        return CondenseResult(
            run_id=task.run_id,
            input_xyz=str(task.input_xyz),
            output_xyz=str(task.output_xyz),
            stride=task.stride,
            input_bytes=task.input_xyz.stat().st_size,
            output_bytes=task.output_xyz.stat().st_size,
            total_frames=total_frames,
            kept_frames=kept_frames,
            status="ok",
            message="",
        )
    except Exception as exc:
        try:
            tmp_output.unlink(missing_ok=True)
        except OSError:
            pass
        return CondenseResult(
            run_id=task.run_id,
            input_xyz=str(task.input_xyz),
            output_xyz=str(task.output_xyz),
            stride=task.stride,
            input_bytes=task.input_xyz.stat().st_size if task.input_xyz.exists() else 0,
            output_bytes=0,
            total_frames=total_frames,
            kept_frames=kept_frames,
            status="error",
            message=str(exc),
        )


def write_manifest(path: Path, results: list[CondenseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "status",
        "stride",
        "total_frames",
        "kept_frames",
        "input_bytes",
        "output_bytes",
        "input_xyz",
        "output_xyz",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.run_id):
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def run_tasks(tasks: list[CondenseTask], workers: int) -> list[CondenseResult]:
    results: list[CondenseResult] = []
    if workers == 1:
        for task in tasks:
            result = condense_one(task)
            results.append(result)
            if result.status == "ok":
                status(
                    f"{task.run_id}: kept {result.kept_frames}/{result.total_frames} frames, "
                    f"{result.output_bytes / 1_000_000_000:.3f} GB"
                )
            else:
                status(f"{task.run_id}: ERROR {result.message}")
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(condense_one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            results.append(result)
            if result.status == "ok":
                status(
                    f"{task.run_id}: kept {result.kept_frames}/{result.total_frames} frames, "
                    f"{result.output_bytes / 1_000_000_000:.3f} GB"
                )
            else:
                status(f"{task.run_id}: ERROR {result.message}")
    return results


def main() -> int:
    args = parse_args()
    tasks = discover_tasks(args.input_root, args.output_root, args.stride, args.run_id)
    status(f"Input root:  {args.input_root}")
    status(f"Output root: {args.output_root}")
    status(f"Stride:      {args.stride}")
    status(f"Workers:     {args.workers}")
    status(f"Trajectories: {len(tasks)}")

    if args.dry_run:
        for task in tasks:
            status(f"DRY RUN {task.run_id}: {task.input_xyz} -> {task.output_xyz}")
        return 0

    results = run_tasks(tasks, args.workers)
    manifest_path = args.output_root / "B1_condense_manifest.csv"
    write_manifest(manifest_path, results)
    status(f"Manifest: {manifest_path}")

    failed = [result for result in results if result.status != "ok"]
    if failed:
        status(f"Failed trajectories: {len(failed)}",)
        return 1

    total_in = sum(result.input_bytes for result in results)
    total_out = sum(result.output_bytes for result in results)
    status(f"Total input:  {total_in / 1_000_000_000:.3f} GB")
    status(f"Total output: {total_out / 1_000_000_000:.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
