#!/usr/bin/env python3
"""Extract the OMOL input archive without modifying or processing its contents."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DEFAULT_ARCHIVE = (
    MLIP_DIR / "outputsfull" / "C1_omol" / "omol25_evaluation_inputs_250915.tar.gz"
)
DEFAULT_OUTPUT_DIR = (
    MLIP_DIR / "outputsfull" / "C1_omol" / "omol25_evaluation_inputs_250915"
)


def safe_member_path(output_dir: Path, member_name: str) -> Path:
    target = (output_dir / member_name).resolve()
    output_root = output_dir.resolve()
    if target != output_root and output_root not in target.parents:
        raise RuntimeError(f"Refusing unsafe archive path outside output dir: {member_name}")
    return target


def extract_archive(archive_path: Path, output_dir: Path, overwrite: bool) -> int:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = 0

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            target = safe_member_path(output_dir, member.name)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                raise RuntimeError(f"Refusing non-regular archive member: {member.name}")

            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {target}. Pass --overwrite to replace files."
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {member.name}")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted_files += 1

    return extracted_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_files = extract_archive(args.archive, args.output_dir, args.overwrite)
    print(f"Extracted {n_files} file(s) from {args.archive} to {args.output_dir}")


if __name__ == "__main__":
    main()
