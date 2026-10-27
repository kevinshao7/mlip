#!/usr/bin/env python3
"""Launch P31 fine tuning from the best completed cycle2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


SAVED_METRIC = re.compile(
    r"\|\s*(?P<split>train|valid)_[^|]*\|\s*"
    r"(?P<energy>[0-9.eE+-]+)\s*\|\s*(?P<force>[0-9.eE+-]+)\s*\|",
    re.IGNORECASE,
)


ROOT = Path(__file__).resolve().parent
MLIP = ROOT.parents[1]
RUN_ID = "cycle2"
PRESSURE_GPA = "31"
DEFAULT_LR = 1e-4
REGISTRY = MLIP / "codes/finetuning_megadataset.jsonl"
TRAINING = (
    MLIP
    / "outputsfull/X_semiactivelearning"
    / RUN_ID
    / "cycles/P31p0000GPa/training"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_prepared_experiment() -> None:
    split_path = TRAINING / "random_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for raw_path, expected_digest in split["acquired_sha256"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_digest:
            raise RuntimeError(f"Prepared DFT snapshot changed: {path}")


def select_best_parent() -> dict[str, object]:
    """Select the lowest completed saved-model validation force RMSE."""
    if not REGISTRY.is_file():
        raise FileNotFoundError(f"Fine-tuning registry is missing: {REGISTRY}")
    candidates = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        context = record.get("campaign_context", {})
        metrics = record.get("metrics", {}).get("final", {})
        if record.get("status") != "completed" or context.get("run_id") != RUN_ID:
            continue
        force = metrics.get("force_rmse_mev_per_a")
        if not isinstance(force, (int, float)) or not math.isfinite(float(force)):
            continue
        run_name = str(record.get("run_name", ""))
        model = Path(str(record.get("run_id", ""))) / "models" / f"{run_name}.model"
        if not model.is_file():
            continue
        endpoint = {}
        logs = sorted((Path(str(record.get("run_id", ""))) / "logs").glob("*.log"))
        text = logs[-1].read_text(errors="replace") if logs else ""
        for match in SAVED_METRIC.finditer(text):
            endpoint[match.group("split").lower()] = {
                "energy_rmse_mev_per_atom": float(match.group("energy")),
                "force_rmse_mev_per_a": float(match.group("force")),
            }
        ranking_metrics = endpoint.get("valid", metrics)
        hyperparameters = record.get("hyperparameters", {})
        try:
            learning_rate = float(hyperparameters["lr"])
        except (KeyError, TypeError, ValueError):
            learning_rate = DEFAULT_LR
        candidates.append({"model": model.resolve(), "run_name": run_name,
                           "cycle": context.get("cycle", "unknown"),
                           "metrics": ranking_metrics, "epoch_metrics": metrics,
                           "learning_rate": learning_rate,
                           "run_directory": Path(str(record.get("run_id", ""))).resolve()})
    if not candidates:
        raise RuntimeError(f"No completed cycle2 fine-tuned model is available in {REGISTRY}")
    return min(candidates, key=lambda item: (
        float(item["metrics"].get("force_rmse_mev_per_a", float("inf"))),
        float(item["metrics"].get("loss", float("inf"))),
        float(item["metrics"].get("energy_rmse_mev_per_atom", float("inf"))),
    ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-lr", "--lr", "--learning-rate", type=float, default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR:g}). Example: -lr 1e-2",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be a finite positive number")
    return args


def build_command(args: argparse.Namespace, parent: Path) -> list[str]:
    if args.lr == DEFAULT_LR:
        changed_parameter = "best_completed_parent"
        change_note = (
            "Start the next 20-epoch P31 batch from the completed cycle2 model "
            "with the lowest recorded validation force RMSE; retain lr=1e-4."
        )
    else:
        changed_parameter = "learning_rate"
        change_note = (
            f"Human-selected learning rate {args.lr:g} replaces the 1e-4 default; "
            "all other prepared P31 settings remain unchanged."
        )
    return [
        sys.executable,
        str(ROOT / "campaign.py"),
        "finetune",
        "--pressure",
        PRESSURE_GPA,
        "--run",
        "--parent-model",
        str(parent),
        "--lr",
        str(args.lr),
        "--changed-parameter",
        changed_parameter,
        "--change-note",
        change_note,
    ]


def main() -> None:
    args = parse_args()
    verify_prepared_experiment()
    best = select_best_parent()
    metrics = best["metrics"]
    print(
        "Selected best completed cycle2 parent by validation force RMSE: "
        f"{best['run_name']} ({best['cycle']}, "
        f"force={metrics.get('force_rmse_mev_per_a')} meV/A, "
        f"energy={metrics.get('energy_rmse_mev_per_atom')} meV/atom, "
        f"loss={metrics.get('loss')})",
        flush=True,
    )
    environment = os.environ.copy()
    environment["SAL_RUN_ID"] = RUN_ID
    command = build_command(args, best["model"])
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
