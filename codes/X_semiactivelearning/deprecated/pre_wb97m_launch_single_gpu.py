#!/usr/bin/env python3
"""Launch the prepared P10 acquisition baseline; no flags required."""
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

os.environ["SAL_RUN_ID"] = "acquisition_p10"
import campaign


def main():
    package = campaign.OUT / "hpc_dataset"
    manifest_path = package / "dataset.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing portable dataset: {manifest_path}; run campaign.py prepare-acquisition locally, then transfer the hpc_dataset directory via GitHub")
    manifest = json.loads(manifest_path.read_text())
    if not manifest["ready_for_training"]:
        raise ValueError(f"Dataset at {package} is a preview of an incomplete transfer; rerun preparation after copying completes")
    if manifest["validation_fraction"] != 0.2:
        raise ValueError("Prepared baseline must use an 80/20 train/validation split")
    for name, digest in manifest["files"].items():
        if campaign.sha(package / name) != digest:
            raise ValueError(f"Portable dataset changed: {name}")
    if not (campaign.OUT / "model_lineage.json").exists():
        campaign.command_init(Namespace())
    cfg = campaign.config()
    cfg["finetune"]["validation_fraction"] = 0.2
    campaign.write_json(campaign.OUT / "campaign_config.json", cfg)
    labelled = campaign.cycle(10.) / "dft/acquisition/processed.extxyz"
    labelled.parent.mkdir(parents=True, exist_ok=True)
    if labelled.exists() and campaign.sha(labelled) != manifest["files"]["processed.extxyz"]:
        raise ValueError("Existing campaign acquisition differs from portable dataset; use a new campaign namespace")
    if not labelled.exists():
        shutil.copy2(package / "processed.extxyz", labelled)
    print(f"Fine-tuning dataset: {package}", flush=True)
    print(f"Frozen split: {manifest['train_count']} train / {manifest['validation_count']} validation", flush=True)
    os.environ["XDG_CACHE_HOME"] = str(campaign.MLIP / "outputsfull/.cache")
    args = campaign.parser().parse_args([
        "finetune", "--pressure", "10", "--run", "--parent-model", "polar-1-l",
        "--changed-parameter", "acquisition_dataset",
        "--change-note", "Approved P10 acquisition, 80/20 split; polar-1-l baseline, lr=1e-4, 20 epochs, no SWA."])
    args.func(args)


if __name__ == "__main__":
    main()
