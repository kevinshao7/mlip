#!/usr/bin/env python3
"""Stream a small, labeled extxyz replay set from an OMol25 tar.gz archive.

OMol25 is distributed as ASE DB compatible LMDB shards.  This program extracts
only one shard at a time to a temporary file, reads its ASE Atoms rows, and
writes a size-bounded extxyz file suitable for MACE-style replay training.
"""

from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
import time
import zlib
from pathlib import Path

from ase import Atoms
from ase.db.row import AtomsRow
from ase.io import write
from ase.io.jsonio import decode


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = Path(r"C:\Users\shaoq\Downloads\train_4M.tar.gz")
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "omol25_replay.extxyz"
MIB = 1024 * 1024


def log(message: str) -> None:
    """Print a timestamped message immediately."""
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def lmdb_value(transaction, key: str):
    raw = transaction.get(key.encode("ascii"))
    if raw is None:
        return None
    return decode(zlib.decompress(raw).decode("utf-8"))


def iter_lmdb_atoms(path: Path, progress_every: int):
    """Yield Atoms from one ASE-LMDB shard using only ase and lmdb."""
    import lmdb

    log(f"Opening temporary LMDB shard: {path.name}")
    env = lmdb.open(
        os.fspath(path), subdir=False, readonly=True, lock=False, readahead=False
    )
    try:
        with env.begin() as transaction:
            next_id = int(lmdb_value(transaction, "nextid") or 1)
            deleted = set(lmdb_value(transaction, "deleted_ids") or [])
            log(f"Shard index reports {next_id - 1:,} row IDs ({len(deleted):,} deleted)")
            for row_id in range(1, next_id):
                if row_id in deleted:
                    continue
                row_dict = lmdb_value(transaction, str(row_id))
                if row_dict is not None:
                    yield row_id, AtomsRow(row_dict).toatoms(
                        add_additional_information=True
                    )
                if row_id % progress_every == 0:
                    log(f"Scanned {row_id:,}/{next_id - 1:,} row IDs in current shard")
    finally:
        env.close()


def prepare_atoms(atoms: Atoms, source_member: str, row_id: int) -> Atoms:
    """Normalize labels and metadata for multi-head/MACE replay training."""
    result = atoms.copy()
    nested = atoms.info.get("data", {})
    if isinstance(nested, dict):
        result.info.update(nested)
    result.info.pop("data", None)

    if atoms.calc is None:
        raise ValueError("OMol25 row has no calculator labels")
    result.info["REF_energy"] = float(atoms.get_potential_energy())
    result.arrays["REF_forces"] = atoms.get_forces().copy()
    result.calc = None
    result.info.setdefault("charge", 0)
    result.info.setdefault("spin", 1)
    result.info["config_type"] = "OMOL25_REPLAY"
    result.info["omol_archive_member"] = source_member
    result.info["omol_row_id"] = row_id
    return result


def copy_member_to_temp(member_file, temp_path: Path, member_size: int) -> None:
    copied = 0
    next_report = 32 * MIB
    with temp_path.open("wb") as destination:
        while True:
            chunk = member_file.read(8 * MIB)
            if not chunk:
                break
            destination.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                log(f"Materialized shard: {copied / MIB:.1f}/{member_size / MIB:.1f} MiB")
                next_report += 32 * MIB
    log(f"Shard ready: {copied / MIB:.1f} MiB")


def extract(args: argparse.Namespace) -> tuple[int, int, int]:
    if not args.archive.is_file():
        raise FileNotFoundError(f"Archive not found: {args.archive}")
    if args.max_output_mb <= 0 or args.max_configs <= 0:
        raise ValueError("--max-output-mb and --max-configs must be positive")

    byte_limit = int(args.max_output_mb * MIB)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = args.temp_dir or args.output.parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {args.output}")
    staging.unlink(missing_ok=True)

    written = skipped = shards = 0
    log(f"Input archive: {args.archive} ({args.archive.stat().st_size / MIB:.1f} MiB)")
    log(f"Output: {args.output}")
    log(f"Hard output limit: {args.max_output_mb:.1f} MiB; config limit: {args.max_configs:,}")

    try:
        # Streaming mode avoids reading/decompressing the complete 20 GB archive index.
        with tarfile.open(args.archive, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".aselmdb"):
                    continue
                shards += 1
                log(f"Found shard #{shards}: {member.name} ({member.size / MIB:.1f} MiB)")
                source = archive.extractfile(member)
                if source is None:
                    log("WARNING: could not open member; skipping")
                    continue
                fd, temp_name = tempfile.mkstemp(
                    prefix="omol25_", suffix=".aselmdb", dir=temp_dir
                )
                os.close(fd)
                temp_path = Path(temp_name)
                try:
                    copy_member_to_temp(source, temp_path, member.size)
                    for row_id, atoms in iter_lmdb_atoms(temp_path, args.progress_every):
                        try:
                            frame = prepare_atoms(atoms, member.name, row_id)
                        except Exception as exc:
                            skipped += 1
                            log(f"WARNING: skipped row {row_id}: {type(exc).__name__}: {exc}")
                            continue

                        before = staging.stat().st_size if staging.exists() else 0
                        write(staging, frame, format="extxyz", append=written > 0)
                        after = staging.stat().st_size
                        if after > byte_limit:
                            with staging.open("r+b") as output_file:
                                output_file.truncate(before)
                            log("Reached output size limit; removed the frame that crossed it")
                            break
                        written += 1
                        if written == 1 or written % args.progress_every == 0:
                            log(
                                f"Wrote {written:,} configs, {after / MIB:.2f} MiB, "
                                f"latest={len(frame)} atoms, skipped={skipped:,}"
                            )
                        if written >= args.max_configs:
                            log("Reached configuration limit")
                            break
                finally:
                    temp_path.unlink(missing_ok=True)
                    log(f"Removed temporary shard: {temp_path.name}")

                current_size = staging.stat().st_size if staging.exists() else 0
                if written >= args.max_configs or current_size >= byte_limit * 0.999:
                    break

        if written == 0:
            raise RuntimeError("No labeled configurations were written")
        os.replace(staging, args.output)
        final_size = args.output.stat().st_size
        log(f"DONE: {written:,} configs from {shards} shard(s), {final_size / MIB:.2f} MiB")
        log(f"Skipped rows: {skipped:,}")
        return written, final_size, skipped
    except Exception:
        log(f"Extraction failed; partial output retained at {staging}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-output-mb", type=float, default=250.0,
        help="Hard cap on final extxyz size in MiB (default: 250).",
    )
    parser.add_argument("--max-configs", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    extract(args)


if __name__ == "__main__":
    main()
