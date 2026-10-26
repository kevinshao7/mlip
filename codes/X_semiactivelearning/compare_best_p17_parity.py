#!/usr/bin/env python3
"""Compare the best completed P17 campaign model and Polar-1-s against P17 DFT.

This is a thin provenance wrapper around D2_naive/plot_finetuning_parity.py.
It selects the lowest completed validation loss from the git-tracked
fine-tuning megadataset and evaluates both models on every current (not
historical snapshot) P17 ``processed.extxyz`` acquisition dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read, write

ROOT = Path(__file__).resolve().parent
MLIP = ROOT.parents[1]
REGISTRY = MLIP / "codes" / "finetuning_megadataset.jsonl"
P17 = 17.7827941004


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def p17_record(run_id: str) -> dict:
    candidates = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        context = record.get("campaign_context", {})
        metrics = record.get("metrics", {}).get("final", {})
        if record.get("status") != "completed" or context.get("run_id") != run_id:
            continue
        if context.get("cycle") != "P17p7828GPa" or not isinstance(metrics.get("loss"), (int, float)):
            continue
        model = Path(record["run_id"]) / "models" / f"{record['run_name']}.model"
        if model.is_file():
            candidates.append((float(metrics["loss"]), float(metrics.get("force_rmse_mev_per_a", float("inf"))),
                               float(metrics.get("energy_rmse_mev_per_atom", float("inf"))), record, model.resolve()))
    if not candidates:
        raise RuntimeError(f"No completed P17 model for run {run_id} in {REGISTRY}")
    _, _, _, record, model = min(candidates, key=lambda item: item[:3])
    return {"record": record, "model": model, "candidate_count": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="cycle2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out_root = MLIP / "outputsfull" / ROOT.name / args.run_id
    processed = sorted((out_root / "cycles" / "P17p7828GPa" / "dft").glob("acquisition*/processed.extxyz"))
    if not processed:
        raise FileNotFoundError("No current P17 processed.extxyz DFT datasets")
    frames = [frame for path in processed for frame in read(path, ":")]
    if not frames:
        raise RuntimeError("P17 DFT datasets contain no validated structures")
    best = p17_record(args.run_id)
    output = out_root / "cycles" / "P17p7828GPa" / "diagnostic" / "best_model_vs_foundation_parity"
    output.mkdir(parents=True, exist_ok=True)
    configs = output / "P17_all_current_validated_dft.extxyz"
    write(configs, frames, format="extxyz")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pressure_gpa": P17, "run_id": args.run_id,
        "selection": "lowest completed validation loss; force RMSE then energy RMSE tie-breakers",
        "best_model": str(best["model"]), "best_registry_record": best["record"],
        "candidate_count": best["candidate_count"], "foundation_model": "polar-1-s",
        "dft_inputs": [{"path": str(path), "sha256": digest(path), "frames": len(read(path, ":"))} for path in processed],
        "total_frames": len(frames), "merged_dft_sha256": digest(configs),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [sys.executable, str(MLIP / "codes" / "D2_naive" / "plot_finetuning_parity.py"),
               "--configs", str(configs), "--finetuned-model", str(best["model"]),
               "--foundation-model", "polar-1-s", "--output-dir", str(output), "--device", args.device]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"Wrote comparable foundation and fine-tuned P17 parity plots to {output}", flush=True)


if __name__ == "__main__":
    main()
