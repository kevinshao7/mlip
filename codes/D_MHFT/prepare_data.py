#!/usr/bin/env python3
"""Prepare target and replay data for PolarMACE multihead fine tuning."""

from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from collections.abc import Iterable
from pathlib import Path

from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read, write

from orca_to_extxyz import DEFAULT_INPUT_DIR, convert_outputs


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_TARGET_ALL = DATA_DIR / "target_all.xyz"
DEFAULT_TARGET_TRAIN = DATA_DIR / "target_train.xyz"
DEFAULT_TARGET_VALID = DATA_DIR / "target_valid.xyz"
DEFAULT_TARGET_TEST = DATA_DIR / "target_test.xyz"
DEFAULT_E0S = DATA_DIR / "target_e0s.json"
DEFAULT_REPLAY = DATA_DIR / "omol_replay_unlabeled.xyz"
DEFAULT_ATOMIZATION = (
    MLIP_DIR / "codes" / "7_7b_clustervalidation" / "atomizationenergies.txt"
)
DEFAULT_OMOL_ARCHIVE = (
    MLIP_DIR / "outputsfull" / "C1_omol" / "omol25_evaluation_inputs_250915.tar.gz"
)


def load_atomization_e0s(path: Path) -> dict[int, float]:
    e0s: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("atom"):
            continue
        symbol, value = [field.strip() for field in line.split(",", maxsplit=1)]
        e0s[int(atomic_numbers[symbol])] = float(value)
    if not e0s:
        raise ValueError(f"No E0 values parsed from {path}")
    return e0s


def write_e0_json(atomization_path: Path, output_path: Path) -> dict[int, float]:
    e0s = load_atomization_e0s(atomization_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({str(key): value for key, value in sorted(e0s.items())}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return e0s


def split_extxyz(
    input_path: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path | None,
    valid_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    frames = read(input_path, index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    frames = list(frames)
    if not frames:
        raise ValueError(f"No frames found in {input_path}")

    n_total = len(frames)
    n_test = int(round(n_total * test_fraction)) if test_path else 0
    n_valid = int(round(n_total * valid_fraction))
    if n_total >= 3:
        if valid_fraction > 0:
            n_valid = max(1, n_valid)
        if test_path and test_fraction > 0:
            n_test = max(1, n_test)
    if n_valid + n_test >= n_total:
        raise ValueError(
            f"Split fractions leave no training data: total={n_total}, "
            f"valid={n_valid}, test={n_test}"
        )

    valid = frames[:n_valid]
    test = frames[n_valid : n_valid + n_test]
    train = frames[n_valid + n_test :]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    write(train_path, train, format="extxyz")
    write(valid_path, valid, format="extxyz")
    if test_path:
        write(test_path, test, format="extxyz")
    return {"train": len(train), "valid": len(valid), "test": len(test), "total": n_total}


def iter_atoms_from_object(obj) -> Iterable[Atoms]:
    if isinstance(obj, Atoms):
        yield obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("initial_atoms"), Atoms):
            yield obj["initial_atoms"]
        else:
            for value in obj.values():
                yield from iter_atoms_from_object(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from iter_atoms_from_object(value)


def extract_unlabeled_omol_replay(
    archive_path: Path,
    output_path: Path,
    max_configs: int,
    config_type: str,
) -> int:
    frames: list[Atoms] = []
    with tarfile.open(archive_path) as archive:
        members = sorted(
            (member for member in archive.getmembers() if member.isfile()),
            key=lambda member: member.name,
        )
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            obj = pickle.load(extracted)
            for atoms in iter_atoms_from_object(obj):
                converted = atoms.copy()
                converted.calc = None
                converted.set_pbc(False)
                converted.info.setdefault("charge", 0)
                converted.info.setdefault("spin", 1)
                converted.info.setdefault("external_field", [0.0, 0.0, 0.0])
                converted.info["config_type"] = config_type
                converted.info["source_file"] = member.name
                frames.append(converted)
                if len(frames) >= max_configs:
                    break
            if len(frames) >= max_configs:
                break

    if not frames:
        raise ValueError(f"No ASE Atoms objects found in {archive_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, frames, format="extxyz")
    return len(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orca-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--target-all", type=Path, default=DEFAULT_TARGET_ALL)
    parser.add_argument("--target-train", type=Path, default=DEFAULT_TARGET_TRAIN)
    parser.add_argument("--target-valid", type=Path, default=DEFAULT_TARGET_VALID)
    parser.add_argument("--target-test", type=Path, default=DEFAULT_TARGET_TEST)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--atomization-energies", type=Path, default=DEFAULT_ATOMIZATION)
    parser.add_argument("--e0s-json", type=Path, default=DEFAULT_E0S)
    parser.add_argument("--omol-archive", type=Path, default=DEFAULT_OMOL_ARCHIVE)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--max-replay-configs", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8, help="Parallel ORCA parser processes.")
    parser.add_argument("--skip-orca", action="store_true")
    parser.add_argument("--skip-omol-replay", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of writing a partial dataset if any ORCA output fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    e0s = write_e0_json(args.atomization_energies, args.e0s_json)
    print(f"Wrote E0s for atomic numbers {sorted(e0s)} to {args.e0s_json}")

    if not args.skip_orca:
        orca_args = argparse.Namespace(
            inputs=[args.orca_dir],
            output=args.target_all,
            all_steps=False,
            charge=None,
            multiplicity=None,
            config_type="ORCA_DFT",
            allow_incomplete=args.allow_incomplete,
            strict=args.strict,
            vacuum=0.0,
            workers=args.workers,
        )
        convert_outputs(orca_args)

    counts = split_extxyz(
        args.target_all,
        args.target_train,
        args.target_valid,
        args.target_test,
        args.valid_fraction,
        args.test_fraction,
    )
    print(
        "Split target data: "
        f"train={counts['train']} valid={counts['valid']} "
        f"test={counts['test']} total={counts['total']}"
    )

    if not args.skip_omol_replay:
        n_replay = extract_unlabeled_omol_replay(
            args.omol_archive,
            args.replay_output,
            args.max_replay_configs,
            "OMOL25_REPLAY_UNLABELED",
        )
        print(f"Wrote {n_replay} unlabeled replay configs to {args.replay_output}")


if __name__ == "__main__":
    main()
