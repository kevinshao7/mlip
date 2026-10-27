#!/usr/bin/env python3
"""Append-only, Git-tracked registry for every MLIP fine-tuning attempt."""
from __future__ import annotations

import hashlib
import fcntl
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MLIP = Path(__file__).resolve().parents[1]
REGISTRY = MLIP / "codes" / "finetuning_megadataset.jsonl"
SCHEMA = 1
METRIC = re.compile(r"(?:Initial|Epoch\s+(?P<epoch>\d+)):\s+head:.*?loss=\s*(?P<loss>[0-9.eE+-]+),\s*RMSE_E_per_atom=\s*(?P<energy>[0-9.eE+-]+)\s*meV,\s*RMSE_F=\s*(?P<force>[0-9.eE+-]+)\s*meV\s*/\s*A")


def digest(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_description(value: str | None) -> dict:
    path = Path(value).resolve() if value else None
    result = {"path": str(path) if path else None, "sha256": digest(path) if path else None}
    if not path or not path.is_file(): return result
    try:
        from ase.io import iread
        sizes = [len(atoms) for atoms in iread(path)]
        result.update({"structures": len(sizes), "atoms_total": sum(sizes), "atoms_min": min(sizes, default=0), "atoms_max": max(sizes, default=0)})
    except Exception as exc: result["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return result


def campaign_context(arguments: dict) -> dict | None:
    """Expose named semi-active-loop lineage when a data path carries it."""
    for value in arguments.values():
        if not isinstance(value, str) or "X_semiactivelearning" not in value: continue
        parts = Path(value).parts
        try:
            marker = parts.index("X_semiactivelearning")
            result = {"output_root": "/".join(parts[:marker + 1])}
            if marker + 1 < len(parts) and parts[marker + 1] != "cycles": result["run_id"] = parts[marker + 1]
            if "cycles" in parts: result["cycle"] = parts[parts.index("cycles") + 1]
            return result
        except (ValueError, IndexError): pass
    return None


def metrics(run_dir: Path) -> dict:
    logs = sorted((run_dir / "logs").glob("*.log")) if (run_dir / "logs").is_dir() else []
    logs = [path for path in logs if not path.name.endswith("_debug.log")]
    if not logs: return {"log": None, "initial": None, "final": None}
    matches = []
    for line in logs[-1].read_text(errors="replace").splitlines():
        hit = METRIC.search(line)
        if hit: matches.append({"epoch": -1 if hit.group("epoch") is None else int(hit.group("epoch")), "loss": float(hit.group("loss")), "energy_rmse_mev_per_atom": float(hit.group("energy")), "force_rmse_mev_per_a": float(hit.group("force"))})
    return {"log": str(logs[-1]), "initial": next((x for x in matches if x["epoch"] == -1), None), "final": matches[-1] if matches else None}


def existing_ids(text: str | None = None) -> set[str]:
    if text is None and not REGISTRY.is_file(): return set()
    ids = set()
    source = REGISTRY.read_text(encoding="utf-8") if text is None else text
    for line in source.splitlines():
        try: ids.add(json.loads(line)["run_id"])
        except (json.JSONDecodeError, KeyError): pass
    return ids


def record(metadata: dict, *, method: str, wrapper: str, status: str, ended_at: str | None = None, duration_seconds: float | None = None, error: str | None = None) -> bool:
    run_dir = Path(metadata["run_directory"]).resolve()
    run_id = str(run_dir)
    arguments = metadata.get("arguments", {})
    observed_metrics = metrics(run_dir)
    if status != "completed" and observed_metrics.get("final") is not None:
        observed_metrics["last_observed"] = observed_metrics["final"]
        observed_metrics["final"] = None
    entry = {"schema_version": SCHEMA, "run_id": run_id, "run_name": metadata.get("run_name", run_dir.name),
             "status": status, "method": method, "wrapper": wrapper, "started_at": metadata.get("created_at"),
             "ended_at": ended_at, "duration_seconds": duration_seconds, "git_revision": metadata.get("git_revision"),
             "foundation_model": arguments.get("foundation_model"), "hyperparameters": arguments,
             "mace_command": metadata.get("mace_command", []), "data_provenance": {
                 "train": data_description(arguments.get("train_file")), "validation": data_description(arguments.get("valid_file")),
                 "replay": data_description(arguments.get("pt_train_file")) if arguments.get("pt_train_file") else None},
             "campaign_context": campaign_context(arguments),
             "metrics": observed_metrics, "error": error}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        if run_id in existing_ids(handle.read()):
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
        handle.seek(0, 2)
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def record_metadata_file(metadata_path: Path, *, method: str, wrapper: str, status: str = "completed", ended_at: str | None = None, duration_seconds: float | None = None, error: str | None = None) -> bool:
    return record(json.loads(metadata_path.read_text(encoding="utf-8")), method=method, wrapper=wrapper, status=status,
                  ended_at=ended_at or datetime.now(timezone.utc).isoformat(), duration_seconds=duration_seconds, error=error)
