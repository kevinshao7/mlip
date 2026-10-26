#!/usr/bin/env python3
"""Backfill the shared fine-tuning registry from immutable run metadata."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import finetune_megadataset as registry

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(); parser.add_argument("--reset", action="store_true", help="Rebuild only the generated registry file.")
args = parser.parse_args()
if args.reset and registry.REGISTRY.exists(): registry.REGISTRY.unlink()
count = 0
for metadata in sorted(ROOT.glob("*/runs/**/run_metadata.json")):
    method = "target_only_naive" if metadata.parts[-4] == "D2_naive" else "multihead_replay"
    run_dir = metadata.parent; logs = [p for p in sorted((run_dir / "logs").glob("*.log")) if not p.name.endswith("_debug.log")]
    text = logs[-1].read_text(errors="replace") if logs else ""
    completed = "INFO: Done" in text
    ended = datetime.fromtimestamp(logs[-1].stat().st_mtime, timezone.utc).isoformat() if completed and logs else None
    started = datetime.fromisoformat(json.loads(metadata.read_text(encoding="utf-8"))["created_at"])
    duration = (datetime.fromisoformat(ended) - started).total_seconds() if ended else None
    status = "completed" if completed else "incomplete_or_unknown"
    if registry.record_metadata_file(metadata, method=method, wrapper=str(metadata.parts[-4]) + "/trainmace.py", status=status, ended_at=ended, duration_seconds=duration):
        count += 1
print(f"Backfilled {count} run(s) into {registry.REGISTRY}")
