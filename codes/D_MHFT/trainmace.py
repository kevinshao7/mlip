#!/usr/bin/env python3
"""Run PolarMACE polar-1-s multihead fine tuning from prepared extxyz files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MLIP_DIR = SCRIPT_DIR.parents[1]
MACE_REPO = MLIP_DIR / "mace"
CACHE_DIR = MLIP_DIR / "outputsfull" / ".cache"
DATA_DIR = SCRIPT_DIR / "data"
RUNS_DIR = SCRIPT_DIR / "runs"


def path_arg(path: Path) -> str:
    return str(path.resolve())


def require_file(path: Path, message: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{message}: {path}")
    return path_arg(path)


def replay_is_labeled(path: Path) -> bool:
    from ase.io import read

    atoms = read(path, index=0)
    return "REF_energy" in atoms.info and "REF_forces" in atoms.arrays


def build_train_argv(args: argparse.Namespace) -> list[str]:
    train_file = require_file(args.train_file, "Missing target training extxyz")
    valid_file = require_file(args.valid_file, "Missing target validation extxyz")
    e0s = require_file(args.e0s, "Missing E0 JSON")
    pt_train_file = require_file(args.pt_train_file, "Missing replay extxyz")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    checkpoints_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    work_dir = run_dir / "work"
    for path in (model_dir, checkpoints_dir, log_dir, results_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)

    pseudolabel_replay = args.pseudolabel_replay
    if pseudolabel_replay == "auto":
        pseudolabel_replay = "False" if replay_is_labeled(args.pt_train_file) else "True"
        print(
            f"Replay labels detected: pseudolabel_replay={pseudolabel_replay} "
            f"for {args.pt_train_file}"
        )

    argv = [
        "run_train.py",
        f"--name={args.name}",
        "--model=PolarMACE",
        f"--foundation_model={args.foundation_model}",
        f"--train_file={train_file}",
        f"--valid_file={valid_file}",
        f"--pt_train_file={pt_train_file}",
        "--multiheads_finetuning=True",
        f"--pseudolabel_replay={pseudolabel_replay}",
        f"--num_samples_pt={args.num_samples_pt}",
        f"--subselect_pt={args.subselect_pt}",
        f"--filter_type_pt={args.filter_type_pt}",
        f"--weight_pt_head={args.weight_pt_head}",
        f"--atomic_numbers={args.atomic_numbers}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--total_charge_key=charge",
        "--total_spin_key=spin",
        f"--E0s={e0s}",
        f"--energy_weight={args.energy_weight}",
        f"--forces_weight={args.forces_weight}",
        "--stress_weight=0.0",
        "--loss=weighted",
        f"--batch_size={args.batch_size}",
        f"--valid_batch_size={args.valid_batch_size}",
        f"--max_num_epochs={args.max_num_epochs}",
        f"--seed={args.seed}",
        f"--default_dtype={args.default_dtype}",
        f"--device={args.device}",
        f"--num_workers={args.num_workers}",
        f"--model_dir={path_arg(model_dir)}",
        f"--checkpoints_dir={path_arg(checkpoints_dir)}",
        f"--log_dir={path_arg(log_dir)}",
        f"--results_dir={path_arg(results_dir)}",
        f"--work_dir={path_arg(work_dir)}",
    ]
    if args.distributed:
        argv.append("--distributed")
        argv.append(f"--launcher={args.launcher}")
    if args.restart_latest:
        argv.append("--restart_latest")
    if args.force_mh_ft_lr:
        argv.append("--force_mh_ft_lr=True")
        argv.append(f"--lr={args.lr}")
    if args.ema:
        argv.append("--ema")
        argv.append(f"--ema_decay={args.ema_decay}")
    if args.swa:
        argv.append("--swa")
        argv.append(f"--start_swa={args.start_swa}")
        argv.append(f"--swa_energy_weight={args.swa_energy_weight}")
        argv.append(f"--swa_forces_weight={args.swa_forces_weight}")
    if args.dry_run:
        print(" ".join(argv))
        raise SystemExit(0)
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="polar1s_mhft_orca")
    parser.add_argument("--foundation-model", default="polar-1-s")
    parser.add_argument("--train-file", type=Path, default=DATA_DIR / "target_train.xyz")
    parser.add_argument("--valid-file", type=Path, default=DATA_DIR / "target_valid.xyz")
    parser.add_argument("--pt-train-file", type=Path, default=DATA_DIR / "omol_replay_unlabeled.xyz")
    parser.add_argument("--e0s", type=Path, default=DATA_DIR / "target_e0s.json")
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR / "polar1s_mhft_orca")
    parser.add_argument("--atomic-numbers", default="[1, 7, 8, 16]")
    parser.add_argument(
        "--pseudolabel-replay",
        default="auto",
        choices=["auto", "True", "False", "true", "false"],
        help="Auto uses original OMol25 labels when present, otherwise foundation-model pseudolabels.",
    )
    parser.add_argument("--num-samples-pt", type=int, default=10000)
    parser.add_argument("--subselect-pt", default="random", choices=["random", "fps"])
    parser.add_argument("--filter-type-pt", default="combinations", choices=["none", "combinations", "inclusive", "exclusive"])
    parser.add_argument("--weight-pt-head", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--valid-batch-size", type=int, default=1)
    parser.add_argument("--max-num-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--default-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--launcher", default="torchrun", choices=["slurm", "torchrun", "mpi", "none"])
    parser.add_argument("--restart-latest", action="store_true")
    parser.add_argument("--force-mh-ft-lr", action="store_true")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.99999)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--start-swa", type=int, default=15)
    parser.add_argument("--swa-energy-weight", type=float, default=10.0)
    parser.add_argument("--swa-forces-weight", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MACE_REPO.is_dir():
        raise FileNotFoundError(f"Local MACE checkout is missing: {MACE_REPO}")

    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))
    sys.path.insert(0, str(MACE_REPO.resolve()))
    sys.argv = build_train_argv(args)

    from mace.cli.run_train import main as run_train_main

    run_train_main()


if __name__ == "__main__":
    main()
