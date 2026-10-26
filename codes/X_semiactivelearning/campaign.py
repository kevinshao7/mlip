#!/usr/bin/env python3
"""Auditable semi-active-learning pressure campaign for PolarMACE.

All state-changing commands append an event to the global JSONL provenance log.
Run ``python campaign.py --help`` for the five operational stage commands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MLIP = ROOT.parents[1]
BASE_OUT = MLIP / "outputsfull" / ROOT.name
RUN_ID = os.environ.get("SAL_RUN_ID", "").strip()
if RUN_ID and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", RUN_ID)):
    raise ValueError("SAL_RUN_ID must contain only letters, digits, '.', '_' or '-'")
OUT = BASE_OUT / RUN_ID if RUN_ID else BASE_OUT
DEFAULT_CONFIG = ROOT / "campaign_config.json"
FORMAL_CHARGES = {"H": 1, "N": -3, "O": -2, "S": -2}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision() -> str:
    result = subprocess.run(["git", "-C", str(MLIP), "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def config_path() -> Path:
    # A named run owns a copied configuration, so later adjustments are not
    # silently inherited from another cycle.  The historical unnamed run
    # retains the repository-level configuration for compatibility.
    return OUT / "campaign_config.json" if RUN_ID and (OUT / "campaign_config.json").is_file() else DEFAULT_CONFIG


def config() -> dict:
    return json.loads(config_path().read_text(encoding="utf-8"))


def relpath(value: str) -> Path:
    return (ROOT / value).resolve()


def tag(pressure: float) -> str:
    return f"P{pressure:.4f}GPa".replace(".", "p")


def cycle(pressure: float) -> Path:
    return OUT / "cycles" / tag(pressure)


def archive_stage(path: Path, pressure: float, stage: str) -> Path | None:
    """Move a superseded stage into immutable cycle history; never delete it."""
    if not path.exists():
        return None
    history = cycle(pressure) / "history" / stage
    history.mkdir(parents=True, exist_ok=True)
    destination = history / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = 1
    while destination.exists():
        suffix += 1; destination = history / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{suffix:02d}"
    shutil.move(str(path), str(destination))
    event("stage_archived", pressure, [], [], archived_stage=stage, archive=str(destination))
    print(f"[history] archived prior {stage} to {destination}", flush=True)
    return destination


def event(stage: str, pressure: float | None, inputs: list[Path] = [], outputs: list[Path] = [], **extra: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": now(), "stage": stage, "pressure_gpa": pressure,
        "git_revision": git_revision(), "python": sys.executable,
        "inputs": [{"path": str(p.resolve()), "sha256": sha(p)} for p in inputs if p.is_file()],
        "outputs": [{"path": str(p.resolve()), "sha256": sha(p)} for p in outputs if p.is_file()], **extra,
    }
    with (OUT / "provenance.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    summary = OUT / "campaign_summary.csv"
    new = not summary.exists()
    with summary.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "pressure_gpa", "stage", "parent_model", "model", "kind"])
        if new: writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in writer.fieldnames})
    timeline = OUT / "campaign_timeline.csv"
    new_timeline = not timeline.exists()
    with timeline.open("a", newline="", encoding="utf-8") as f:
        fields = ["timestamp_utc", "pressure_gpa", "stage", "kind", "parent_model", "model", "archive", "inputs", "outputs"]
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_timeline: writer.writeheader()
        writer.writerow({"timestamp_utc":record["timestamp_utc"], "pressure_gpa":pressure, "stage":stage,
                         "kind":record.get("kind", ""), "parent_model":record.get("parent_model", ""),
                         "model":record.get("model", ""), "archive":record.get("archive", ""),
                         "inputs":";".join(item["path"] for item in record["inputs"]),
                         "outputs":";".join(item["path"] for item in record["outputs"])})


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_profile(pressure: float) -> tuple[float, float]:
    cfg = config()
    rows = []
    with relpath(cfg["profile_csv"]).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = float(row["pressure_GPa"])
            if 1 <= p <= 100 and all(abs(p - old[0]) > 1e-9 for old in rows):
                rows.append((p, float(row["density_g_cm3"]), float(row[cfg["profile_temperature_column"]])))
    rows.sort()
    if pressure < rows[0][0] or pressure > rows[-1][0]:
        raise ValueError(f"Pressure {pressure:g} GPa is outside profile range 1--100 GPa")
    for p, rho, temp in rows:
        if abs(p - pressure) < 1e-9:
            return rho, temp
    import math
    for (p0, rho0, t0), (p1, rho1, t1) in zip(rows, rows[1:]):
        if p0 < pressure < p1:
            frac = (math.log10(pressure) - math.log10(p0)) / (math.log10(p1) - math.log10(p0))
            return rho0 + frac * (rho1-rho0), t0 + frac * (t1-t0)
    raise AssertionError("unreachable profile interpolation")


def current_parent() -> str:
    state = OUT / "model_lineage.json"
    if state.exists():
        return json.loads(state.read_text())["current_model"]
    return config()["finetune"]["initial_parent"]


def completed_campaign_models(pressure: float) -> list[dict]:
    """Read completed models for this named campaign from the append-only registry."""
    registry = MLIP / "codes" / "finetuning_megadataset.jsonl"
    if not registry.is_file():
        raise FileNotFoundError(f"Fine-tuning registry is missing: {registry}")
    selected = []
    for line in registry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        context, metrics = record.get("campaign_context", {}), record.get("metrics", {}).get("final", {})
        cycle_name = str(context.get("cycle", ""))
        match = re.fullmatch(r"P([0-9]+(?:p[0-9]+)?)GPa", cycle_name)
        cycle_pressure = float(match.group(1).replace("p", ".")) if match else float("nan")
        # Campaign cycle tags retain four decimal places, so compare against
        # the requested floating-point pressure at tag precision.
        if (record.get("status") != "completed" or context.get("run_id", "") != (RUN_ID or "") or
                abs(cycle_pressure - pressure) > 5e-4):
            continue
        run_root = Path(record.get("run_id", ""))
        model = run_root / "models" / f"{record.get('run_name', '')}.model"
        if model.is_file() and isinstance(metrics.get("loss"), (int, float)):
            selected.append({"record": record, "model": model.resolve(), "metrics": metrics})
    if not selected:
        raise RuntimeError(f"No completed, registry-recorded model is available for {pressure:g} GPa in run {RUN_ID or 'default'}")
    return selected


def command_promote_best(args: argparse.Namespace) -> None:
    """Pin the best completed checkpoint as the parent for a new MD stage."""
    candidates = completed_campaign_models(args.pressure)
    metric_keys = {
        "validation_loss": ("loss", "force_rmse_mev_per_a", "energy_rmse_mev_per_atom"),
        "force_rmse": ("force_rmse_mev_per_a", "loss", "energy_rmse_mev_per_atom"),
        "energy_rmse": ("energy_rmse_mev_per_atom", "loss", "force_rmse_mev_per_a"),
    }
    keys = metric_keys[args.metric]
    best = min(candidates, key=lambda item: tuple(float(item["metrics"].get(key, float("inf"))) for key in keys))
    state_path = OUT / "model_lineage.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    previous = state.get("current_model", config()["finetune"]["initial_parent"])
    selection = {
        "timestamp_utc": now(), "source_pressure_gpa": args.pressure,
        "selection_metric": args.metric, "selection_keys": list(keys),
        "previous_current_model": previous, "selected_model": str(best["model"]),
        "selected_run_id": best["record"]["run_id"], "selected_run_name": best["record"]["run_name"],
        "selected_final_metrics": best["metrics"], "candidate_count": len(candidates),
        "registry": str((MLIP / "codes" / "finetuning_megadataset.jsonl").resolve()),
    }
    state.setdefault("promotions", []).append(selection)
    state["current_model"] = str(best["model"])
    write_json(state_path, state)
    selection_path = cycle(args.pressure) / "parent_selection.json"
    write_json(selection_path, selection)
    event("model_promoted_for_next_pressure", args.pressure,
          [MLIP / "codes" / "finetuning_megadataset.jsonl", best["model"]],
          [state_path, selection_path], parent_model=previous, model=str(best["model"]),
          selection_metric=args.metric, selected_final_metrics=best["metrics"],
          selected_run_name=best["record"]["run_name"], candidate_count=len(candidates))
    print(f"Promoted {best['model']} by {args.metric}: {best['metrics']}", flush=True)


def active_branch(state: dict) -> str:
    """Return the immutable lineage branch currently eligible for promotion."""
    return str(state.get("active_branch", "legacy"))


def branch_history(state: dict, branch: str | None = None) -> list[dict]:
    """Select history for one branch; pre-branch records are ``legacy``."""
    selected = active_branch(state) if branch is None else branch
    return [entry for entry in state.get("history", [])
            if str(entry.get("branch_id", "legacy")) == selected]


def command_init(_: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    active_config = OUT / "campaign_config.json" if RUN_ID else DEFAULT_CONFIG
    if RUN_ID and not active_config.exists():
        shutil.copy2(DEFAULT_CONFIG, active_config)
    shutil.copy2(active_config, OUT / "campaign_config.initial.json")
    state = OUT / "model_lineage.json"
    if not state.exists():
        write_json(state, {"current_model": "polar-1-s", "root_model": "polar-1-s",
                           "active_branch": "foundation_initial", "branches": [], "history": []})
    event("initialized", None, [active_config], [OUT / "campaign_config.initial.json", state], campaign=config()["campaign_name"], run_id=RUN_ID or "default")
    print(f"Initialized {OUT} (run={RUN_ID or 'default'})")


def inherited_dft_paths() -> list[Path]:
    """Return immutable snapshots intentionally inherited by this named run."""
    manifest = OUT / "inherited_dft" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    paths = []
    for record in data.get("datasets", []):
        path = Path(record["copied_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Inherited DFT snapshot is missing: {path}")
        if sha(path) != record["copied_sha256"]:
            raise RuntimeError(f"Inherited DFT snapshot changed unexpectedly: {path}")
        paths.append(path)
    return paths


def command_inherit_dft(args: argparse.Namespace) -> None:
    """Snapshot validated labels from an earlier run into this new run only."""
    if not RUN_ID:
        raise ValueError("DFT inheritance requires a named run, e.g. SAL_RUN_ID=cycle2")
    source_root = (BASE_OUT / args.from_run).resolve() if args.from_run else BASE_OUT.resolve()
    if source_root == OUT.resolve():
        raise ValueError("Source and destination runs must differ")
    if not (source_root / "model_lineage.json").is_file():
        raise FileNotFoundError(f"Not a completed source campaign run: {source_root}")
    sources = sorted(source_root.glob("cycles/*/dft/acquisition*/processed.extxyz"))
    if not sources:
        raise FileNotFoundError(f"No validated acquisition DFT labels found in {source_root}")
    destination_root = OUT / "inherited_dft"
    manifest = destination_root / "manifest.json"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing inherited DFT manifest: {manifest}")
    records = []
    for source in sources:
        relative = source.relative_to(source_root)
        copied = destination_root / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        records.append({"source_path": str(source.resolve()), "source_sha256": sha(source),
                        "copied_path": str(copied.resolve()), "copied_sha256": sha(copied)})
    write_json(manifest, {"created_utc": now(), "source_run": str(source_root),
                          "source_run_id": args.from_run or "default", "datasets": records})
    event("dft_inherited_snapshot", None, sources, [manifest, *[Path(row["copied_path"]) for row in records]],
          run_id=RUN_ID, source_run=args.from_run or "default", dataset_count=len(records))
    print(f"Snapshotted {len(records)} validated DFT datasets into {destination_root}", flush=True)


def command_reuse_dft(args: argparse.Namespace) -> None:
    """Reuse safe unfinished inputs and the event ledger in a named run.

    Normally terminated outputs are already represented by ``inherit-dft``.
    Explicit SCF/ORCA failures stay skipped.  Only never-started inputs and
    outputs interrupted without an error marker are copied into a new,
    Cycle2-owned DFT stage.
    """
    if not RUN_ID:
        raise ValueError("DFT reuse requires a named run, e.g. SAL_RUN_ID=cycle2")
    source_root = (BASE_OUT / args.from_run).resolve() if args.from_run else BASE_OUT.resolve()
    source_cycle = source_root / "cycles" / tag(args.pressure)
    source_dft = source_cycle / "dft" / args.source_kind
    source_manifest = source_dft / "manifest.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing source DFT manifest: {source_manifest}")
    target_kind = args.target_kind
    if not re.fullmatch(r"acquisition_r[2-9][0-9]*", target_kind):
        raise ValueError("--target-kind must be acquisition_r<N> with N >= 2")
    target_cycle = cycle(args.pressure); target_dft = target_cycle / "dft" / target_kind
    if target_dft.exists():
        raise FileExistsError(f"Refusing to overwrite existing Cycle2 DFT stage: {target_dft}")
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    safe, skipped = [], []
    for row in source_rows:
        stem = row["stem"]; normal = source_dft / "outputs" / f"{stem}.out"
        partial = source_dft / "outputs" / f"{stem}.out.partial"
        if local_orca_complete(normal):
            skipped.append({"stem": stem, "status": "already_normal_output"})
        elif partial.exists() and classify_orca_output(partial) != "unknown":
            skipped.append({"stem": stem, "status": "skipped_prior_scf_or_orca_failure",
                            "classification": classify_orca_output(partial)})
        else:
            row = dict(row)
            row["reuse_status"] = "interrupted_without_error" if partial.exists() else "not_started"
            safe.append(row)
    if not safe:
        raise RuntimeError("No safe unfinished DFT inputs remain to reuse")
    target_inputs = target_dft / "inputs"; target_inputs.mkdir(parents=True)
    for row in safe:
        source_input = source_dft / row["input"]
        if not source_input.is_file():
            raise FileNotFoundError(f"Source input is missing: {source_input}")
        copied = target_inputs / Path(row["input"]).name
        shutil.copy2(source_input, copied)
        row["input"] = str(copied.relative_to(target_dft))
    with (target_dft / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(safe[0].keys()))
        writer.writeheader(); writer.writerows(safe)
    # The copied summary is the explicit provenance boundary: new Cycle2 work
    # never writes into the earlier run's DFT directories.
    reuse_summary = target_dft / "reuse_summary.json"
    write_json(reuse_summary, {"created_utc": now(), "source_run": str(source_root),
                               "source_dft": str(source_dft), "source_kind": args.source_kind,
                               "target_kind": target_kind, "reused": safe, "skipped": skipped,
                               "policy": "normal outputs inherited; SCF/ORCA failures skipped; only cleanly interrupted or never-started inputs resumed"})
    # Preserve selection exclusion state and analysis source, so a later
    # acquisition_r4 can deterministically choose fresh components from the
    # same production trajectory rather than rediscovering these 50 events.
    source_selection = source_cycle / "clusters" / args.source_kind / "selection.csv"
    source_summary = source_cycle / "analysis" / "summary.json"
    source_events = source_cycle / "analysis" / "events.csv"
    source_approval = source_cycle / "approval_acquisition.json"
    for source in (source_selection, source_summary, source_events, source_approval):
        if not source.is_file():
            raise FileNotFoundError(f"Cannot enable fresh-cluster fallback; missing source artifact: {source}")
    selection_target = target_cycle / "clusters" / args.source_kind / "selection.csv"
    selection_target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source_selection, selection_target)
    analysis_target = target_cycle / "analysis"; analysis_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_summary, analysis_target / "summary.json"); shutil.copy2(source_events, analysis_target / "events.csv")
    shutil.copy2(source_approval, target_cycle / "approval_acquisition.json")
    event("dft_reuse_prepared", args.pressure,
          [source_manifest, source_selection, source_summary, source_events, source_approval],
          [target_dft / "manifest.csv", reuse_summary, selection_target, analysis_target / "summary.json", analysis_target / "events.csv"],
          run_id=RUN_ID, source_run=args.from_run or "default", source_kind=args.source_kind,
          target_kind=target_kind, reused=len(safe), skipped=len(skipped),
          fresh_cluster_fallback=f"campaign.py clusters --pressure {args.pressure:g} --kind acquisition_r4")
    print(f"Prepared {len(safe)} reusable DFT inputs in {target_dft}; {len(skipped)} prior outputs/failures excluded.", flush=True)


def command_status(_: argparse.Namespace) -> None:
    state = OUT / "model_lineage.json"
    lineage = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {"current_model": "not initialized"}
    rows = []
    for pressure in config()["pressures_gpa"]:
        cdir = cycle(pressure)
        rows.append({"pressure_gpa": pressure, "md_prepared": (cdir / "md" / "manifest.json").is_file(), "md_completed": bool(list((cdir / "md").glob("production_*.xyz"))), "analysis": (cdir / "analysis" / "summary.json").is_file(), "approved": (cdir / "approval_acquisition.json").is_file(), "acquisition_dft": (cdir / "dft" / "acquisition" / "processed.extxyz").is_file()})
    print(json.dumps({"current_model": lineage["current_model"], "active_branch": active_branch(lineage),
                      "active_branch_batches": len(branch_history(lineage)), "cycles": rows,
                      "provenance_log": str(OUT / "provenance.jsonl"),
                      "timeline": str(OUT / "campaign_timeline.csv"),
                      "model_lineage": str(OUT / "model_lineage.json")}, indent=2))


def command_restart_foundation(args: argparse.Namespace) -> None:
    """Start a fresh training branch from PolarMACE while retaining all history."""
    state_path = OUT / "model_lineage.json"
    if not state_path.exists():
        raise FileNotFoundError("Campaign is not initialized; run `python campaign.py init` first")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    previous_branch = active_branch(state)
    previous_model = state.get("current_model", config()["finetune"]["initial_parent"])
    # Existing entries predate branch support. They remain immutable and are
    # explicitly marked legacy rather than being discarded or overwritten.
    for entry in state.get("history", []):
        entry.setdefault("branch_id", "legacy")
    new_branch = "foundation_restart_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = OUT / "lineage_branches" / f"{new_branch}_previous_state.json"
    write_json(archive, state)
    state.setdefault("branches", []).append({
        "branch_id": new_branch, "utc": now(), "parent_branch": previous_branch,
        "parent_model": previous_model, "root_model": "polar-1-s",
        "reason": args.reason, "previous_state": str(archive),
    })
    state["active_branch"] = new_branch
    state["current_model"] = "polar-1-s"
    write_json(state_path, state)
    event("lineage_restarted_from_foundation", None, [archive], [state_path],
          branch_id=new_branch, previous_branch=previous_branch,
          previous_model=previous_model, new_parent="polar-1-s", reason=args.reason)
    print(f"Started immutable branch {new_branch} from polar-1-s; prior lineage retained at {archive}", flush=True)


def replace_assignment(text: str, name: str, value: str) -> str:
    text, n = re.subn(rf"^{re.escape(name)}\s*=.*$", f"{name} = {value}", text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"Could not patch {name} in MD base script")
    return text


def command_md(args: argparse.Namespace) -> None:
    cfg, cdir = config(), cycle(args.pressure)
    md = cdir / "md"
    if md.exists() and any(md.iterdir()):
        archive_stage(md, args.pressure, "md")
    md.mkdir(parents=True, exist_ok=True)
    source = relpath(cfg["md"]["base_script"])
    rho, temp = read_profile(args.pressure)
    parent = current_parent()
    text = source.read_text(encoding="utf-8")
    text = replace_assignment(text, "PROJECT_ROOT", repr(str(MLIP)))
    text = replace_assignment(text, "MD_RESULTS_DIR", repr(str(md)))
    text = replace_assignment(text, "target_profile_densitygcm3", repr(rho))
    text = replace_assignment(text, "pressuregpa", repr(args.pressure))
    text = replace_assignment(text, "T_final", repr(temp))
    text = replace_assignment(text, "targetmolecules", repr(cfg["md"]["target_molecules"]))
    text = replace_assignment(text, "moleculemass", repr((18.01528 + 17.03052) / 2))
    pressure_steps = round(cfg["md"]["pressure_ramp_ps"] * 1000 / cfg["md"]["timestep_fs"])
    temperature_steps = round(cfg["md"]["temperature_ramp_ps"] * 1000 / cfg["md"]["timestep_fs"])
    production_steps = round(cfg["md"]["production_duration_ps"] * 1000 / cfg["md"]["timestep_fs"])
    total_steps = pressure_steps + temperature_steps + production_steps
    save_interval = production_steps // cfg["md"]["target_production_frames"]
    if save_interval < 1:
        raise ValueError("target_production_frames is incompatible with MD step count")
    text = replace_assignment(text, "pressureramptime", f"{cfg['md']['pressure_ramp_ps']}*1000*units.fs")
    text = replace_assignment(text, "tempramptime", f"{cfg['md']['temperature_ramp_ps']}*1000*units.fs")
    text = replace_assignment(text, "totaltimesteps", repr(total_steps))
    text = replace_assignment(text, "saveinterval", repr(save_interval))
    text = replace_assignment(text, "composition_label", repr(cfg["composition"]["label"]))
    text = replace_assignment(text, "ammonia_water_ratio", repr(cfg["composition"]["ammonia_water_ratio"]))
    text, n = re.subn(r"^simbox\.add_solvent\(.*$", "simbox.add_solvent([water, amm], ratio=[1, 1], zdim=boxsize, density=densitygcm3)", text, count=1, flags=re.M)
    if n != 1: raise RuntimeError("Could not patch R1 mixture into MD base script")
    text = re.sub(r'model="polar-1-s"', f"model={parent!r}", text, count=1)
    # Save exactly the requested production-hold frames, rather than creating
    # high-volume ramp trajectories that cannot enter the cluster workflow.
    text, n = re.subn(r"next_save_step = s", "next_save_step = pressure_ramp_steps + temperature_ramp_steps + s", text, count=1)
    if n != 1: raise RuntimeError("Could not set production-only first save step")
    text, n = re.subn(r"if total_steps >= next_save_step:", "if stage == \"production_hold\" and total_steps >= next_save_step:", text, count=1)
    if n != 1: raise RuntimeError("Could not restrict trajectory saving to production hold")
    generated = md / "run_md.py"; generated.write_text(text, encoding="utf-8")
    launcher = md / "run_md.sh"
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + f"export CUDA_VISIBLE_DEVICES={cfg['md']['gpu']}\nexec {shlex.quote(sys.executable)} run_md.py\n", encoding="utf-8")
    launcher.chmod(0o755)
    manifest = {"pressure_gpa": args.pressure, "density_g_cm3": rho, "temperature_K": temp, "parent_model": parent, "seed_policy": "one random seed", "runtime_target_hours": cfg["md"]["walltime_target_hours"], "pressure_ramp_ps": cfg["md"]["pressure_ramp_ps"], "temperature_ramp_ps": cfg["md"]["temperature_ramp_ps"], "production_duration_ps": cfg["md"]["production_duration_ps"], "total_steps": total_steps, "production_steps": production_steps, "save_interval_steps": save_interval, "target_production_frames": cfg["md"]["target_production_frames"], "generated": now()}
    write_json(md / "manifest.json", manifest)
    event("md_prepared", args.pressure, [source, config_path()], [generated, launcher, md / "manifest.json"], parent_model=parent)
    print(f"Prepared {generated}. Run: {launcher}")
    if args.run:
        subprocess.run([str(launcher)], cwd=md, check=True)
        trajectories = sorted(md.glob("production_*.xyz"))
        event("md_completed", args.pressure, [generated], trajectories, parent_model=parent)


def load_species_module():
    sys.path.insert(0, str((ROOT / "../B_conditionsproduction").resolve()))
    import plot_final_species_distribution as species
    return species


def fast_molecular_components(frame, species):
    """Vectorized equivalent of B's molecular_components() for large MD scans."""
    import numpy as np
    from ase.geometry import find_mic
    natoms = len(frame.symbols)
    displacement = frame.positions[:, None, :] - frame.positions[None, :, :]
    _vectors, flat = find_mic(displacement.reshape(-1, 3), cell=frame.cell, pbc=True)
    distances = flat.reshape(natoms, natoms)
    adjacency = [set() for _ in range(natoms)]
    for left in range(natoms - 1):
        if frame.symbols[left] == "H": continue
        for right in range(left + 1, natoms):
            if frame.symbols[right] == "H": continue
            cutoff = species.pair_cutoff(frame.symbols[left], frame.symbols[right], 1.45, 1.30, 0.50, 1.20)
            if distances[left, right] <= cutoff:
                adjacency[left].add(right); adjacency[right].add(left)
    eligible = np.array([i for i, symbol in enumerate(frame.symbols) if symbol in {"H", "O", "N"}], dtype=int)
    for hydrogen, symbol in enumerate(frame.symbols):
        if symbol != "H": continue
        candidates = eligible[eligible != hydrogen]
        if candidates.size:
            nearest = int(candidates[int(np.argmin(distances[hydrogen, candidates]))])
            adjacency[hydrogen].add(nearest); adjacency[nearest].add(hydrogen)
    unseen=set(range(natoms)); components=[]
    while unseen:
        start=unseen.pop(); component=[start]; stack=[start]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unseen:
                    unseen.remove(neighbour); component.append(neighbour); stack.append(neighbour)
        components.append(sorted(component))
    return components


def pressure_gpa_from_comment(comment: str) -> float:
    """Convert ASE/extxyz virial stress (eV/Å³) to hydrostatic pressure in GPa."""
    match = re.search(r'stress="([^"]+)"', comment)
    if match is None:
        return float("nan")
    values = [float(value) for value in match.group(1).split()]
    if len(values) != 9:
        return float("nan")
    # ASE stress uses the mechanics sign convention: p = -trace(stress) / 3.
    return -(values[0] + values[4] + values[8]) / 3 * 160.21766208


def command_analyze(args: argparse.Namespace) -> None:
    from collections import Counter
    import numpy as np
    if args.production_frame_stride < 1:
        raise ValueError("--production-frame-stride must be positive")
    species = load_species_module(); cdir = cycle(args.pressure); md = cdir / "md"
    trajectories = [args.trajectory.resolve()] if args.trajectory else sorted(md.glob("production_*.xyz"))
    if len(trajectories) != 1 or not trajectories[0].is_file(): raise FileNotFoundError(f"Expected exactly one readable trajectory, got {trajectories}")
    frame_time_ps = args.frame_time_ps
    if frame_time_ps is None:
        md_cfg=config()["md"]
        production_steps=round(md_cfg["production_duration_ps"] * 1000 / md_cfg["timestep_fs"])
        frame_time_ps=(production_steps // md_cfg["target_production_frames"]) * md_cfg["timestep_fs"] / 1000
    out = cdir / "analysis"
    if out.exists() and any(out.iterdir()):
        archive_stage(out, args.pressure, "analysis")
    out.mkdir(parents=True, exist_ok=True)
    allowed = set(config()["analysis"]["base_species"]); rows=[]; events=[]; thermo_rows=[]; total_components=0
    for frame_index, frame in species.iter_xyz_frames(trajectories[0]):
        if frame_index < args.production_start_frame:
            continue
        thermo_rows.append({"frame": frame_index, "production_time_ps": (frame_index - args.production_start_frame) * frame_time_ps, "pressure_GPa": pressure_gpa_from_comment(frame.comment), "volume_A3": float(np.linalg.det(frame.cell))})
        if (frame_index - args.production_start_frame) % args.production_frame_stride:
            continue
        components = fast_molecular_components(frame, species)
        total_components += len(components)
        formulas = [species.component_formula(frame, x) for x in components]
        counts = Counter(formulas); rows.append({"frame": frame_index, **counts})
        for index, formula in enumerate(formulas):
            if formula not in allowed:
                events.append({"frame": frame_index, "component": index, "formula": formula})
    columns = sorted({k for row in rows for k in row if k != "frame"})
    with (out / "species_time_series.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=["frame", *columns]); writer.writeheader(); writer.writerows(rows)
    with (out / "events.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=["frame", "component", "formula"]); writer.writeheader(); writer.writerows(events)
    with (out / "pressure_volume_time_series.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=["frame", "production_time_ps", "pressure_GPa", "volume_A3"]); writer.writeheader(); writer.writerows(thermo_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(12, 6))
    for formula in columns:
        axis.plot([row["frame"] for row in rows], [row.get(formula, 0) for row in rows], label=formula, linewidth=1)
    axis.set_yscale("symlog", linthresh=1); axis.set_xlabel("saved MD frame"); axis.set_ylabel("complete molecular-component count")
    axis.set_title(f"Species evolution, {args.pressure:g} GPa, R1; non-base events are DFT candidates")
    axis.legend(fontsize="small", ncol=2); figure.tight_layout(); figure.savefig(out / "species_time_series.png", dpi=180); plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    times=np.asarray([row["production_time_ps"] for row in thermo_rows]); pressures=np.asarray([row["pressure_GPa"] for row in thermo_rows]); volumes=np.asarray([row["volume_A3"] for row in thermo_rows])
    axes[0].plot(times, pressures, linewidth=.8, color="#2563eb"); axes[0].axhline(args.pressure, color="black", linestyle="--", label=f"target {args.pressure:g} GPa"); axes[0].set(ylabel="hydrostatic pressure / GPa", title="Production-regime equilibration diagnostic"); axes[0].legend()
    axes[1].plot(times, volumes, linewidth=.8, color="#dc2626"); axes[1].set(xlabel="production time / ps", ylabel="cell volume / Å³")
    figure.tight_layout(); figure.savefig(out / "pressure_volume_time_series.png", dpi=180); plt.close(figure)
    illegal_fraction = len(events) / total_components if total_components else 0.0
    max_fraction=config()["analysis"]["max_illegal_component_fraction"]
    eligible=bool(events) and illegal_fraction < max_fraction
    pressure_finite=pressures[np.isfinite(pressures)]
    summary={"trajectory": str(trajectories[0]), "production_start_frame_0based": args.production_start_frame, "production_frame_stride": args.production_frame_stride, "thermo_frames": len(thermo_rows), "frame_time_ps": frame_time_ps, "mean_pressure_GPa": float(np.mean(pressure_finite)) if pressure_finite.size else None, "pressure_std_GPa": float(np.std(pressure_finite)) if pressure_finite.size else None, "volume_start_A3": float(volumes[0]) if volumes.size else None, "volume_end_A3": float(volumes[-1]) if volumes.size else None, "frames": len(rows), "base_species": sorted(allowed), "non_base_events": len(events), "complete_components_sampled": total_components, "illegal_component_fraction": illegal_fraction, "max_illegal_component_fraction": max_fraction, "cluster_eligible": eligible, "non_base_formulas": sorted({x['formula'] for x in events}), "human_review_required": bool(events) and not eligible}
    write_json(out / "summary.json", summary)
    event("analysis_completed", args.pressure, [trajectories[0]], [out / "species_time_series.csv", out / "species_time_series.png", out / "pressure_volume_time_series.csv", out / "pressure_volume_time_series.png", out / "events.csv", out / "summary.json"], non_base_events=len(events), illegal_component_fraction=illegal_fraction, cluster_eligible=eligible)
    print(json.dumps(summary, indent=2))


def command_approve(args: argparse.Namespace) -> None:
    if not args.reason.strip(): raise ValueError("--reason must state the human scientific basis for acquisition")
    path = cycle(args.pressure) / "approval_acquisition.json"
    if path.exists(): raise FileExistsError(f"Approval already exists: {path}")
    write_json(path, {"approved_utc": now(), "pressure_gpa": args.pressure, "reason": args.reason, "guide_energy_error_mev_per_atom": 50, "automatic_threshold": False})
    event("acquisition_approved", args.pressure, [cycle(args.pressure) / "analysis" / "summary.json"], [path], reason=args.reason)
    print(f"Acquisition approved: {path}")


def command_clusters(args: argparse.Namespace) -> None:
    import numpy as np
    from ase import Atoms
    from ase.geometry import find_mic
    from ase.io import write
    cdir = cycle(args.pressure); analysis = cdir / "analysis"
    acquisition = args.kind.startswith("acquisition")
    if acquisition and not (cdir / "approval_acquisition.json").is_file():
        raise PermissionError("Acquisition clusters require a recorded human approval: campaign.py approve")
    selected_count = config()["clusters"]["acquisition_count" if acquisition else "diagnostic_count"]
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("cluster_eligible", False):
        raise ValueError("Cluster selection requires non-base components with an illegal-component fraction below 5%; inspect analysis/summary.json")
    events = list(csv.DictReader((analysis / "events.csv").open(encoding="utf-8")))
    if not events: raise ValueError("No non-base molecular events: no clusters should be selected")
    # Subsequent acquisition rounds exclude event components selected in every
    # preceding acquisition round.  This gives genuinely new configurations
    # from the same immutable production trajectory without overwriting DFT.
    if acquisition and args.kind != "acquisition":
        used = set()
        for previous in cdir.glob("clusters/acquisition*/selection.csv"):
            with previous.open(encoding="utf-8") as stream:
                used.update((row["frame"], row["component"]) for row in csv.DictReader(stream))
        events = [row for row in events if (row["frame"], row["component"]) not in used]
        if not events: raise ValueError("No unselected acquisition events remain for a new DFT round")
        print(f"[{args.kind}] excluded {len(used)} event components selected by earlier acquisition rounds.", flush=True)
    # Evenly sample the remaining chronological event list.
    indices = np.linspace(0, len(events) - 1, min(selected_count, len(events)), dtype=int)
    chosen = [events[i] for i in sorted(set(indices.tolist()))]
    species = load_species_module(); trajectory = Path(summary["trajectory"])
    print(f"[{args.kind}] {len(events)} eligible non-base event(s); selecting {len(chosen)} cluster centre(s).", flush=True)
    print(f"[{args.kind}] Reading source trajectory: {trajectory}", flush=True)
    needed = {int(row["frame"]) for row in chosen}; frames = {}
    for i, frame in species.iter_xyz_frames(trajectory):
        if i in needed: frames[i] = frame
        if i and i % 100 == 0:
            print(f"[{args.kind}] scanned source frame {i}; located {len(frames)}/{len(needed)} selected frame(s).", flush=True)
    clusters = []; metadata = []; visualizations = []; dft_visualizations = []
    for row in chosen:
        frame = frames[int(row["frame"])]
        components, _ = species.molecular_components(frame, 1.45, 1.30, 0.50, 1.20)
        event_component = components[int(row["component"])]
        # A component can straddle a periodic boundary. Anchor all positions on
        # one central atom before computing centres or writing the DFT cluster.
        anchor = frame.positions[event_component[0]]
        central_vectors, _ = find_mic(frame.positions[event_component] - anchor, cell=frame.cell, pbc=True)
        central_center = central_vectors.mean(axis=0)
        distances = []; allowed_centres = {}
        allowed = set(summary["base_species"])
        for idx, component in enumerate(components):
            formula = species.component_formula(frame, component)
            if formula not in allowed:
                continue
            component_vectors, _ = find_mic(frame.positions[component] - anchor, cell=frame.cell, pbc=True)
            vector, distance = find_mic(component_vectors.mean(axis=0) - central_center, cell=frame.cell, pbc=True)
            distances.append((float(distance), idx))
            allowed_centres[idx] = component_vectors.mean(axis=0)
        allowed_sorted = [idx for _distance, idx in sorted(distances)]
        molecule_count = config()["clusters"]["allowed_molecules"]
        if len(allowed_sorted) < molecule_count:
            raise ValueError(f"Frame {row['frame']} has only {len(allowed_sorted)} allowed complete molecules near event; need {molecule_count}")
        # The event is only a diagnosis marker.  Its nearest allowed molecule
        # becomes the DFT-cluster centre; select that centre plus its five
        # nearest allowed molecular neighbours (six complete molecules).
        selected_centre = allowed_sorted[0]
        neighbour_distances = []
        for idx in allowed_sorted[1:]:
            _vector, distance = find_mic(allowed_centres[idx] - allowed_centres[selected_centre], cell=frame.cell, pbc=True)
            neighbour_distances.append((float(distance), idx))
        selected = [selected_centre, *[idx for _distance, idx in sorted(neighbour_distances)[:molecule_count - 1]]]
        atoms_idx = [atom for component in selected for atom in components[component]]
        vectors, _ = find_mic(frame.positions[atoms_idx] - anchor, cell=frame.cell, pbc=True)
        positions = vectors
        atoms = Atoms([frame.symbols[i] for i in atoms_idx], positions=positions, pbc=False)
        selected_formulas = [species.component_formula(frame, components[c]) for c in selected]
        if any(formula not in allowed for formula in selected_formulas):
            raise AssertionError("Illegal component reached a DFT/training cluster")
        atoms.info.update({"source_pressure_gpa": args.pressure, "source_frame": int(row["frame"]), "event_component": int(row["component"]), "event_formula": row["formula"], "event_included_in_dft": False, "central_allowed_component": selected_centre, "central_allowed_formula": selected_formulas[0], "central_atom_count": len(components[selected_centre]), "cluster_kind": args.kind, "molecules": len(selected), "allowed_molecules_only": True})
        visual_components = [int(row["component"]), *allowed_sorted[:3]]
        visual_atom_indices = [atom for component in visual_components for atom in components[component]]
        visual_vectors, _ = find_mic(frame.positions[visual_atom_indices] - anchor, cell=frame.cell, pbc=True)
        visualizations.append({
            "frame": int(row["frame"]), "central_formula": row["formula"],
            "symbols": [frame.symbols[i] for i in visual_atom_indices], "positions": visual_vectors,
            "component_offsets": [sum(len(components[c]) for c in visual_components[:i]) for i in range(4)],
            "component_sizes": [len(components[c]) for c in visual_components],
            "component_formulas": [species.component_formula(frame, components[c]) for c in visual_components],
        })
        dft_visualizations.append({
            "frame": int(row["frame"]), "event_formula": row["formula"],
            "molecules": len(selected),
            "symbols": [frame.symbols[i] for i in atoms_idx], "positions": positions,
            "component_offsets": [sum(len(components[c]) for c in selected[:i]) for i in range(len(selected))],
            "component_sizes": [len(components[c]) for c in selected],
            "component_formulas": selected_formulas,
        })
        preview_formulas = [species.component_formula(frame, components[c]) for c in visual_components[1:]]
        print(f"[{args.kind}] cluster {len(clusters)+1}/{len(chosen)}: frame={row['frame']} event={row['formula']} "
              f"dft_molecules={len(selected)}_allowed_only centre={selected_formulas[0]} neighbours={len(selected)-1} event_excluded=True preview_nearest3={','.join(preview_formulas)} atoms={len(atoms)}", flush=True)
        clusters.append(atoms); metadata.append({**row, "n_molecules": len(selected), "n_atoms": len(atoms)})
    out = cdir / "clusters" / args.kind
    if out.exists() and any(out.iterdir()):
        archive_stage(out, args.pressure, f"clusters_{args.kind}")
    out.mkdir(parents=True, exist_ok=True)
    xyz = out / "clusters.extxyz"; write(xyz, clusters, format="extxyz")
    with (out / "selection.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=metadata[0].keys()); writer.writeheader(); writer.writerows(metadata)
    gallery = write_cluster_gallery(clusters, out)
    videos = write_event_videos(visualizations, out / "event_videos")
    dft_videos = write_dft_cluster_videos(dft_visualizations, out / "dft_cluster_videos")
    event("clusters_selected", args.pressure, [trajectory, analysis / "events.csv"], [xyz, out / "selection.csv", gallery, *videos, *dft_videos], kind=args.kind, count=len(clusters), policy=f"illegal event excluded; one central allowed molecule plus {config()['clusters']['allowed_molecules'] - 1} nearest allowed complete molecular neighbors only")
    print(f"Wrote {len(clusters)} {args.kind} clusters: {xyz}")


def formal_charge(symbols: list[str]) -> int:
    unknown = sorted(set(symbols) - FORMAL_CHARGES.keys())
    if unknown: raise ValueError(f"No formal-charge convention for: {unknown}")
    return sum(FORMAL_CHARGES[x] for x in symbols)


def write_cluster_gallery(clusters, destination: Path) -> Path:
    """Human-review 3D gallery for event-centred complete-molecule clusters."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = 2; rows = (len(clusters) + columns - 1) // columns
    figure = plt.figure(figsize=(11, 4.8 * rows))
    colors = {"H": "#8b8b8b", "O": "#d62728", "N": "#2878b5", "S": "#d4a017"}
    for index, atoms in enumerate(clusters, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        central_count = int(atoms.info.get("central_atom_count", 0))
        for atom_index, (symbol, position) in enumerate(zip(atoms.symbols, atoms.positions, strict=True)):
            central = atom_index < central_count
            axis.scatter(*position, s=90 if central else 28, c="#ff00aa" if central else colors.get(symbol, "black"), edgecolors="black" if central else "none", linewidths=.6 if central else 0, alpha=1 if central else .72)
        label = f"frame {atoms.info.get('source_frame')} | event {atoms.info.get('event_formula')} excluded | {atoms.info.get('molecules')} allowed molecules"
        axis.set_title(label)
        axis.set(xlabel="x / Å", ylabel="y / Å", zlabel="z / Å"); axis.view_init(elev=22, azim=35)
    figure.suptitle("DFT/training clusters: central allowed molecule (magenta) + allowed neighbours; grey H, red O, blue N", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, .98)); path = destination / "event_cluster_gallery.png"; figure.savefig(path, dpi=180); plt.close(figure)
    return path


def write_event_videos(events, destination: Path) -> list[Path]:
    """Render 180-degree GIF pans of only the event plus three nearest molecules."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    destination.mkdir(parents=True, exist_ok=True)
    outputs=[]; summary=[]
    palette=("#ff00aa", "#40a9ff", "#ff7f0e", "#2ca02c")
    for event_data in events:
        print(f"[visualize] rendering 180-degree event animation for production frame {event_data['frame']}", flush=True)
        positions=np.asarray(event_data["positions"]); figure=plt.figure(figsize=(5.5,5.2)); axis=figure.add_subplot(111, projection="3d")
        span=max(float(np.ptp(positions[:, i])) for i in range(3)); half=max(3.0, span*.62); centre=positions.mean(axis=0)
        for component, (offset, size, formula) in enumerate(zip(event_data["component_offsets"], event_data["component_sizes"], event_data["component_formulas"], strict=True)):
            coords=positions[offset:offset+size]
            axis.scatter(coords[:,0], coords[:,1], coords[:,2], s=95 if component==0 else 55, c=palette[component], edgecolors="black", linewidths=.5, label=("event " if component==0 else "nearest ") + formula)
            # Compact intramolecular bonds make the molecule identities legible.
            for left in range(size - 1):
                for right in range(left + 1, size):
                    if np.linalg.norm(coords[left]-coords[right]) < 1.35:
                        axis.plot(*coords[[left,right]].T, color=palette[component], linewidth=2)
        axis.set(xlim=(centre[0]-half, centre[0]+half), ylim=(centre[1]-half, centre[1]+half), zlim=(centre[2]-half, centre[2]+half), xlabel="x / Å", ylabel="y / Å", zlabel="z / Å")
        axis.set_title(f"production frame {event_data['frame']}: {event_data['central_formula']} + 3 nearest complete species")
        axis.legend(loc="upper left", fontsize="small")
        path=destination / f"frame_{event_data['frame']:04d}_{event_data['central_formula']}.gif"
        writer=PillowWriter(fps=12)
        writer.setup(figure, path, dpi=130)
        try:
            for angle in np.linspace(-90, 90, 37):
                axis.view_init(elev=20, azim=float(angle)); writer.grab_frame()
        finally:
            writer.finish()
        plt.close(figure); outputs.append(path)
        print(f"[visualize] wrote {path}", flush=True)
        summary.append({"frame":event_data["frame"], "event":event_data["central_formula"], "nearest_species": ";".join(event_data["component_formulas"][1:]), "file":path.name})
    with (destination / "README.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=["frame","event","nearest_species","file"]); writer.writeheader(); writer.writerows(summary)
    outputs.append(destination / "README.csv")
    return outputs


def write_dft_cluster_videos(clusters, destination: Path) -> list[Path]:
    """Render actual DFT-input clusters, making the allowed-only content auditable."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    destination.mkdir(parents=True, exist_ok=True)
    palette=plt.get_cmap("tab10").colors; outputs=[]; rows=[]
    allowed=set(config()["analysis"]["base_species"])
    for cluster in clusters:
        if any(formula not in allowed for formula in cluster["component_formulas"]):
            raise AssertionError("Refusing to visualize a DFT cluster containing an illegal species")
        positions=np.asarray(cluster["positions"]); figure=plt.figure(figsize=(6.0,5.6)); axis=figure.add_subplot(111, projection="3d")
        span=max(float(np.ptp(positions[:, i])) for i in range(3)); half=max(3.0, span*.62); centre=positions.mean(axis=0)
        for component, (offset, size, formula) in enumerate(zip(cluster["component_offsets"], cluster["component_sizes"], cluster["component_formulas"], strict=True)):
            coords=positions[offset:offset+size]; color=palette[component % len(palette)]
            axis.scatter(coords[:,0], coords[:,1], coords[:,2], s=58, c=[color], edgecolors="black", linewidths=.35, label=f"{component+1}: {formula}")
            for left in range(size - 1):
                for right in range(left + 1, size):
                    if np.linalg.norm(coords[left]-coords[right]) < 1.35:
                        axis.plot(*coords[[left,right]].T, color=color, linewidth=1.8)
        axis.set(xlim=(centre[0]-half, centre[0]+half), ylim=(centre[1]-half, centre[1]+half), zlim=(centre[2]-half, centre[2]+half), xlabel="x / Å", ylabel="y / Å", zlabel="z / Å")
        axis.set_title(f"DFT input, production frame {cluster['frame']}: {cluster['molecules']} allowed molecules\ntrigger {cluster['event_formula']} excluded")
        axis.legend(loc="upper left", fontsize="x-small", ncol=2)
        path=destination / f"frame_{cluster['frame']:04d}_allowed_only.gif"; print(f"[dft-visualize] rendering allowed-only DFT cluster for production frame {cluster['frame']}", flush=True)
        writer=PillowWriter(fps=12); writer.setup(figure, path, dpi=130)
        try:
            for angle in np.linspace(-90, 90, 37):
                axis.view_init(elev=20, azim=float(angle)); writer.grab_frame()
        finally:
            writer.finish()
        plt.close(figure); outputs.append(path); print(f"[dft-visualize] wrote {path}", flush=True)
        rows.append({"frame":cluster["frame"], "excluded_event":cluster["event_formula"], "dft_molecules":cluster["molecules"], "allowed_species": ";".join(cluster["component_formulas"]), "file":path.name})
    with (destination / "README.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=["frame","excluded_event","dft_molecules","allowed_species","file"]); writer.writeheader(); writer.writerows(rows)
    outputs.append(destination / "README.csv")
    return outputs


def dft_input(atoms, charge: int, multiplicity: int, nprocs: int) -> str:
    # def2-TZVPD is declared in the route line.  In ORCA 6, ``NewGTO all``
    # is not valid %basis syntax (it expects a particular atomic number/name),
    # so do not redundantly redeclare the basis block here.
    lines = ["! wB97M-V def2-TZVPD EnGrad RIJCOSX", f"%pal nprocs {nprocs} end", f"* xyz {charge} {multiplicity}"]
    lines += [f"{symbol} {x:.12f} {y:.12f} {z:.12f}" for symbol, (x,y,z) in zip(atoms.symbols, atoms.positions, strict=True)]
    return "\n".join(lines) + "\n*\n"


def write_parity_report(frames, destination: Path) -> Path:
    """Make the human-review diagnostic; collection stays valid if no GPU is available."""
    report = destination / "parity_summary.json"
    if not frames:
        write_json(report, {"status":"unavailable", "reason":"no validated DFT frames", "parent_model":current_parent(), "n_clusters":0})
        return report
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mace.calculators import mace_polar
        model = current_parent()
        calculator = mace_polar(model=model, device="cuda", default_dtype="float64")
        dft_e=[]; mace_e=[]; dft_f=[]; mace_f=[]
        for atoms in frames:
            atoms.calc=calculator
            dft_e.append(float(atoms.info["REF_energy"]) / len(atoms)); mace_e.append(float(atoms.get_potential_energy()) / len(atoms))
            dft_f.extend(atoms.arrays["REF_forces"].ravel()); mace_f.extend(atoms.get_forces().ravel())
        dft_e=np.asarray(dft_e); mace_e=np.asarray(mace_e); dft_f=np.asarray(dft_f); mace_f=np.asarray(mace_f)
        stats={"status":"complete", "parent_model":model, "energy_mae_mev_per_atom":float(np.mean(np.abs(mace_e-dft_e))*1000), "force_mae_ev_per_angstrom":float(np.mean(np.abs(mace_f-dft_f))), "n_clusters":len(frames)}
        figure, axes=plt.subplots(1,2,figsize=(10,4))
        for axis, reference, prediction, label in ((axes[0],dft_e,mace_e,"energy / eV atom$^{-1}$"),(axes[1],dft_f,mace_f,"force component / eV Å$^{-1}$")):
            lower=min(reference.min(),prediction.min()); upper=max(reference.max(),prediction.max()); axis.scatter(reference,prediction,s=14); axis.plot([lower,upper],[lower,upper],"k--"); axis.set(xlabel="DFT "+label,ylabel="MACE "+label)
        figure.suptitle(f"Diagnostic parity: energy MAE {stats['energy_mae_mev_per_atom']:.1f} meV/atom")
        figure.tight_layout(); figure.savefig(destination / "parity.png",dpi=180); plt.close(figure)
    except Exception as exc:
        stats={"status":"unavailable", "reason":f"{type(exc).__name__}: {exc}", "parent_model":current_parent(), "n_clusters":len(frames)}
    write_json(report,stats); return report


def local_orca_complete(output: Path) -> bool:
    """Return whether an ORCA output reached its normal termination marker."""
    if not output.is_file():
        return False
    try:
        return "ORCA TERMINATED NORMALLY" in output.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def classify_orca_output(output: Path) -> str:
    if local_orca_complete(output):
        return "normal"
    if not output.is_file():
        return "not_started"
    text = output.read_text(encoding="utf-8", errors="replace")
    if re.search(r"SCF NOT CONVERGED|ORCA_LEANSCF.*not converged", text, re.I):
        return "scf_nonconverged"
    if re.search(r"mpirun: command not found|error termination in Startup", text, re.I):
        return "mpi_startup"
    if re.search(r"Error:|INPUT ERROR|error termination", text, re.I):
        return "orca_error"
    return "unknown"


def run_local_orca(args: argparse.Namespace, ddir: Path) -> None:
    """Run the prepared manifest serially, with one all-core ORCA calculation at a time."""
    manifest = ddir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Prepare DFT inputs first: {manifest}")
    cfg = config()["dft"]
    configured_command = str(cfg["local_orca_command"])
    resolved_command = shutil.which(configured_command) if Path(configured_command).name == configured_command else configured_command
    if resolved_command is None:
        raise FileNotFoundError(f"Configured local ORCA command is not on PATH: {configured_command}")
    executable = Path(resolved_command).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Configured local ORCA executable is unavailable: {executable}")
    if int(cfg["local_cpu_ranks"]) != int(cfg["orca_nprocs"]):
        raise ValueError("local_cpu_ranks must match orca_nprocs so ORCA input and allocation agree")
    with manifest.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError(f"No DFT jobs listed in {manifest}")
    outputs = ddir / "outputs"; outputs.mkdir(exist_ok=True)
    env = os.environ.copy()
    # The ORCA %pal block supplies MPI ranks. Prevent linked math libraries
    # from spawning a second layer of threads and oversubscribing the node.
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    env["PATH"] = f"{executable.parent}:{env.get('PATH', '')}"
    mpi_module = str(cfg["local_mpi_module"])
    # ``orca_qc`` supplies ORCA itself, while the site MPI launcher is exposed
    # through an environment module.  Use a login bash because ``module`` is a
    # shell function, then exec ORCA so its exit status remains observable.
    mpi_check = subprocess.run(["/bin/bash", "-lc", f"module load {shlex.quote(mpi_module)} && command -v mpirun"],
                               text=True, capture_output=True, env=env)
    if mpi_check.returncode != 0:
        raise RuntimeError(f"Could not load {mpi_module} / find mpirun: {mpi_check.stderr.strip()}")
    mpirun = mpi_check.stdout.strip().splitlines()[-1]
    started = now()
    failure_report = ddir / "dft_failure_report.json"
    try:
        prior_failures = json.loads(failure_report.read_text(encoding="utf-8")).get("failures", [])
    except (FileNotFoundError, json.JSONDecodeError):
        prior_failures = []
    failures = [row for row in prior_failures if row.get("action") == "skipped_no_scf_retry"]
    previously_skipped = {row.get("stem") for row in failures}
    event("dft_local_started", args.pressure, [manifest], [], kind=args.kind,
          executable=str(executable), mpi_module=mpi_module, mpirun=mpirun,
          cpu_ranks=int(cfg["local_cpu_ranks"]), jobs=len(records))
    for index, record in enumerate(records, start=1):
        inp = (ddir / record["input"]).resolve()
        output = outputs / f"{record['stem']}.out"
        if local_orca_complete(output):
            print(f"[local-dft] {index}/{len(records)} {record['stem']}: already complete; skipping", flush=True)
            continue
        if record["stem"] in previously_skipped:
            print(f"[local-dft] {index}/{len(records)} {record['stem']}: previously failed; skipping without retry", flush=True)
            continue
        temporary = output.with_suffix(".out.partial")
        if temporary.exists():
            failure_kind = classify_orca_output(temporary)
            # A user-requested clean stop leaves an incomplete, but not failed,
            # output.  Reused inputs explicitly carry this provenance state and
            # may be restarted with identical ORCA/SCF settings.  All genuine
            # failures remain terminal: do not retry or alter SCF parameters.
            if (failure_kind == "unknown" and
                    record.get("reuse_status") == "interrupted_without_error"):
                interrupted = outputs / "interrupted"
                interrupted.mkdir(exist_ok=True)
                archived = interrupted / f"{record['stem']}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.out.partial"
                temporary.replace(archived)
                print(f"[local-dft] {index}/{len(records)} {record['stem']}: cleanly interrupted; restarting unchanged", flush=True)
            else:
                failures.append({"stem":record["stem"], "exit_code":None, "classification":failure_kind, "output":str(temporary), "action":"skipped_no_scf_retry"})
                previously_skipped.add(record["stem"])
                print(f"[local-dft] {index}/{len(records)} {record['stem']}: existing failed output ({failure_kind}); skipping without retry", flush=True)
                continue
        print(f"[local-dft] {index}/{len(records)} {record['stem']}: starting with {cfg['local_cpu_ranks']} CPU ranks", flush=True)
        with temporary.open("w", encoding="utf-8") as stream:
            launch = f"module load {shlex.quote(mpi_module)} && exec {shlex.quote(str(executable))} {shlex.quote(str(inp))}"
            result = subprocess.run(["/bin/bash", "-lc", launch], cwd=ddir, stdout=stream,
                                    stderr=subprocess.STDOUT, env=env, text=True)
        if result.returncode != 0 or not local_orca_complete(temporary):
            failure_kind = classify_orca_output(temporary)
            failures.append({"stem":record["stem"], "exit_code":result.returncode, "classification":failure_kind, "output":str(temporary), "action":"skipped_no_scf_retry"})
            print(f"[local-dft] {index}/{len(records)} {record['stem']}: skipped ({failure_kind}); retained {temporary}", flush=True)
            continue
        temporary.replace(output)
        print(f"[local-dft] {index}/{len(records)} {record['stem']}: complete", flush=True)
    completed = [outputs / f"{row['stem']}.out" for row in records if local_orca_complete(outputs / f"{row['stem']}.out")]
    write_json(failure_report, {"updated_utc":now(), "policy":"no SCF parameter changes; failed jobs skipped", "failures":failures, "completed":len(completed), "requested":len(records)})
    event("dft_local_completed", args.pressure, [manifest], [*completed, failure_report], kind=args.kind,
          executable=str(executable), cpu_ranks=int(cfg["local_cpu_ranks"]),
          mpi_module=mpi_module, mpirun=mpirun, completed=len(completed), failed=len(failures), requested=len(records), started_utc=started)
    print(f"[local-dft] complete: {len(completed)}/{len(records)} normal; {len(failures)} skipped; report={failure_report}", flush=True)


def command_dft(args: argparse.Namespace) -> None:
    cdir=cycle(args.pressure); ddir=cdir / "dft" / args.kind; clusters=cdir / "clusters" / args.kind / "clusters.extxyz"
    actions=sum(bool(value) for value in (args.submit, args.run_local, args.collect, args.parity))
    if actions > 1:
        raise ValueError("Choose at most one DFT action: --submit, --run-local, --collect, or --parity")
    if args.parity:
        from ase.io import read
        processed=ddir / "processed.extxyz"
        if not processed.is_file(): raise FileNotFoundError("Parity requires collected validated DFT frames")
        old=[ddir / name for name in ("parity.png", "parity_summary.json") if (ddir / name).exists()]
        if old:
            archive=ddir / "history" / "parity" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive.mkdir(parents=True, exist_ok=True)
            for artifact in old: shutil.move(str(artifact), str(archive / artifact.name))
            event("dft_parity_archived", args.pressure, [processed], [], kind=args.kind, archive=str(archive))
        report=write_parity_report(read(processed, ":"), ddir)
        plots=[ddir / "parity.png"] if (ddir / "parity.png").exists() else []
        event("dft_parity_completed", args.pressure, [processed], [report, *plots], kind=args.kind, parent_model=current_parent())
        print(f"Re-evaluated parity: {report}"); return
    if args.collect:
        sys.path.insert(0, str((ROOT / "../C3_DFTproductionstopH2").resolve()))
        import condense_dft_outputs as condenser
        from ase.io import write
        manifest=ddir / "manifest.csv"; outputs=ddir / "outputs"; accepted=[]; rows=[]
        with manifest.open(newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                job=condenser.Job(frame=int(raw["frame"]), stem=raw["stem"], input_path=ddir / raw["input"], output_path=outputs / f"{raw['stem']}.out", manifest_charge=int(raw["charge"]), manifest_spin=int(raw["multiplicity"]), manifest_natoms=int(raw["n_atoms"]))
                result, atoms=condenser.process(job); rows.append(result)
                if atoms is not None: accepted.append(atoms)
        prior_collection = [ddir / name for name in ("processed.extxyz", "collection.csv", "parity.png", "parity_summary.json") if (ddir / name).exists()]
        if prior_collection:
            archive = ddir / "history" / "collection" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive.mkdir(parents=True, exist_ok=True)
            for artifact in prior_collection:
                shutil.move(str(artifact), str(archive / artifact.name))
            event("dft_collection_archived", args.pressure, [], [], kind=args.kind, archive=str(archive))
        processed=ddir / "processed.extxyz"; write(processed, accepted, format="extxyz")
        with (ddir / "collection.csv").open("w", newline="", encoding="utf-8") as f:
            writer=csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row})); writer.writeheader(); writer.writerows(rows)
        parity=write_parity_report(accepted, ddir)
        event("dft_collected", args.pressure, [manifest], [processed, ddir / "collection.csv", parity], kind=args.kind, accepted=len(accepted), requested=len(rows))
        print(f"Collected {len(accepted)}/{len(rows)} validated DFT frames; parity status: {parity}: {processed}"); return
    if args.run_local:
        run_local_orca(args, ddir)
        return
    from ase.io import read
    if not clusters.is_file(): raise FileNotFoundError(f"Generate clusters first: {clusters}")
    if ddir.exists() and any(ddir.iterdir()):
        if not args.replace_prepared:
            raise FileExistsError(f"Refusing to overwrite existing DFT stage: {ddir}. Use --replace-prepared only when no DFT output has run.")
        outputs = ddir / "outputs"
        complete_outputs = [output for output in outputs.glob("*.out") if local_orca_complete(output)] if outputs.exists() else []
        if complete_outputs:
            raise FileExistsError(f"Cannot replace prepared inputs with normally completed ORCA output(s): {complete_outputs[0]}")
        archive = ddir.with_name(f"{ddir.name}_prepared_archived_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        shutil.move(str(ddir), str(archive))
        event("dft_prepared_archived", args.pressure, [archive / "manifest.csv"], [], kind=args.kind, archive=str(archive), reason="replace_prepared_no_normal_orca_outputs")
        print(f"Archived unused or failed prepared inputs to {archive}", flush=True)
    ddir.mkdir(parents=True); inp=ddir / "inputs"; inp.mkdir(); cfg=config()["dft"]
    frames=read(clusters, ":"); records=[]
    for i, atoms in enumerate(frames):
        stem=f"cluster_{i:03d}"; charge=formal_charge(atoms.get_chemical_symbols()); mult=1 if (sum(atoms.numbers)-charge)%2 == 0 else 2
        (inp / f"{stem}.inp").write_text(dft_input(atoms, charge, mult, cfg["orca_nprocs"]), encoding="utf-8")
        records.append({"frame":i,"stem":stem,"input":f"inputs/{stem}.inp","charge":charge,"multiplicity":mult,"n_atoms":len(atoms)})
    with (ddir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    script=ddir / "submit.slurm"
    stems=" ".join(row["stem"] for row in records)
    script.write_text(f"#!/bin/bash\n#SBATCH --job-name=sal_{tag(args.pressure)}_{args.kind}\n#SBATCH --partition={cfg['slurm_partition']}\n#SBATCH --ntasks={cfg['orca_nprocs']}\n#SBATCH --time={cfg['slurm_time']}\nset -euo pipefail\nmodule purge\nmodule load {cfg['orca_module']}\nmkdir -p outputs\nfor stem in {stems}; do {cfg['orca_command']} inputs/${{stem}}.inp > outputs/${{stem}}.out; done\n", encoding="utf-8")
    event("dft_prepared", args.pressure, [clusters], [ddir / "manifest.csv", script], kind=args.kind, cluster_count=len(frames), formal_charge_verified=True)
    print(f"Generated {len(frames)} ORCA inputs and {script}")
    if args.submit:
        if shutil.which("sbatch") is None:
            raise RuntimeError("sbatch is unavailable on this host. Run locally with --run-local, or generate here and submit from a Slurm login node.")
        subprocess.run(["sbatch", str(script)], cwd=ddir, check=True)
        event("dft_submitted", args.pressure, [script], [], kind=args.kind)


def command_finetune(args: argparse.Namespace) -> None:
    from ase.io import read, write
    cdir=cycle(args.pressure); processed=sorted((cdir / "dft").glob("acquisition*/processed.extxyz"))
    inherited = inherited_dft_paths()
    if not processed and not inherited:
        raise FileNotFoundError("Fine tuning requires collected local or immutable inherited acquisition DFT data")
    cfg=config()["finetune"]
    acquired = sorted({*OUT.glob("cycles/*/dft/acquisition*/processed.extxyz"), *inherited})
    if not acquired: raise ValueError("Fine tuning uses only X_semiactivelearning DFT; no collected campaign DFT dataset exists")
    # The pool is exclusively campaign DFT labels.  Do not use the static D2
    # validation file: it has a different chemistry/size distribution and
    # becomes progressively less representative as campaign labels grow.
    all_frames=[]
    for labelled in acquired:
        all_frames.extend(read(labelled, ":"))
    if len(all_frames) < 2:
        raise ValueError("Random 70/30 fine-tuning split requires at least two campaign DFT structures")
    fraction=float(cfg.get("validation_fraction", 0.30))
    if not 0.0 < fraction < 1.0:
        raise ValueError("finetune.validation_fraction must lie strictly between zero and one")
    acquired_sha256={str(x.resolve()):sha(x) for x in acquired}
    # A changing data snapshot deterministically creates a new randomized
    # split; the manifest preserves the seed and exact members for replay.
    snapshot_token="\n".join(f"{path}:{digest}" for path, digest in sorted(acquired_sha256.items()))
    split_seed=int(hashlib.sha256(snapshot_token.encode("utf-8")).hexdigest()[:16], 16)
    indices=list(range(len(all_frames))); random.Random(split_seed).shuffle(indices)
    valid_count=max(1, min(len(all_frames)-1, round(len(all_frames)*fraction)))
    valid_indices=sorted(indices[:valid_count]); train_indices=sorted(indices[valid_count:])
    train=[all_frames[index] for index in train_indices]
    validation=[all_frames[index] for index in valid_indices]
    data=cdir / "training"
    if data.exists() and any(data.iterdir()):
        archive_stage(data, args.pressure, "training")
    data.mkdir(parents=True, exist_ok=True)
    all_data=data / "cumulative_all.extxyz"; merged=data / "cumulative_train.extxyz"; valid=data / "random_validation.extxyz"
    split=data / "random_split.json"
    write(all_data, all_frames, format="extxyz"); write(merged, train, format="extxyz"); write(valid, validation, format="extxyz")
    write_json(split, {"created_utc":now(), "method":"deterministic_shuffle_per_campaign_data_snapshot", "seed":split_seed,
                       "validation_fraction":fraction, "total_frames":len(all_frames), "train_count":len(train), "validation_count":len(validation),
                       "train_indices":train_indices, "validation_indices":valid_indices, "acquired_sha256":acquired_sha256})
    parent=current_parent(); launcher=(ROOT / "../D2_naive/launch_single_gpu.py").resolve()
    lineage=json.loads((OUT / "model_lineage.json").read_text(encoding="utf-8"))
    current_branch = active_branch(lineage)
    batch_number=1 + sum(1 for entry in branch_history(lineage)
                         if float(entry.get("pressure_gpa", -1)) == float(args.pressure))
    finetune_id=f"sal_{tag(args.pressure)}_batch{batch_number:03d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    epochs=args.epochs if args.epochs is not None else int(cfg["batch_epochs"])
    if epochs < 1: raise ValueError("--epochs must be positive")
    # Campaign batches are deliberately short (normally 20 epochs).  Do not
    # enter D2_naive's Stage-Two/SWA recipe at epoch 15: it raises the force
    # loss weight from 100 to 100,000 and the learning rate from 1e-4 to 3e-4
    # for the final five epochs of *every* autonomous batch.  This repeatedly
    # destabilised validation forces.  Long standalone D2 experiments retain
    # their own opt-in SWA defaults; this policy is campaign-specific.
    command=[sys.executable, str(launcher), "--gpu", cfg["gpu"], "--", f"--foundation-model={parent}", f"--train-file={merged}", f"--valid-file={valid}", f"--name={finetune_id}", f"--max-num-epochs={epochs}", "--no-swa", f"--changed-parameter={args.changed_parameter}", f"--change-note={args.change_note}"]
    (data / "launch_command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n", encoding="utf-8")
    event("finetune_prepared", args.pressure, [*acquired], [all_data, merged, valid, split, data / "launch_command.sh"], parent_model=parent, finetune_id=finetune_id, batch_number=batch_number, batch_epochs=epochs, acquired_datasets=[str(x) for x in acquired], training_data_scope="named_run_campaign_DFT_plus_immutable_inherited_snapshot", validation_policy="randomized_30_percent_campaign_only", validation_fraction=fraction, validation_seed=split_seed, train_count=len(train), validation_count=len(validation), changed_parameter=args.changed_parameter, change_note=args.change_note)
    print(shlex.join(command))
    if args.run:
        subprocess.run(command, cwd=launcher.parent, check=True)
        candidates=sorted((launcher.parent / "runs").glob(f"{finetune_id}_*/models/*.model"), key=lambda p:p.stat().st_mtime)
        if not candidates: raise RuntimeError("Fine tuning completed but no campaign-named model was found")
        model=str(candidates[-1].resolve()); state=json.loads((OUT / "model_lineage.json").read_text()); state["history"].append({"pressure_gpa":args.pressure,"parent":parent,"model":model,"train_sha256":sha(merged),"validation_sha256":sha(valid),"split_sha256":sha(split),"acquired_sha256":acquired_sha256,"utc":now(),"batch_number":batch_number,"batch_epochs":epochs,"finetune_id":finetune_id,"branch_id":current_branch,"validation_policy":"randomized_30_percent_campaign_only","validation_fraction":fraction,"validation_seed":split_seed,"train_count":len(train),"validation_count":len(validation)}); state["current_model"]=model; write_json(OUT / "model_lineage.json", state)
        write_json(data / "batch_complete.json", {"utc":now(), "batch_number":batch_number, "batch_epochs":epochs, "parent_model":parent, "model":model, "train_sha256":sha(merged), "validation_sha256":sha(valid), "split_sha256":sha(split), "validation_policy":"randomized_30_percent_campaign_only", "validation_fraction":fraction, "validation_seed":split_seed, "train_count":len(train), "validation_count":len(validation), "acquired_datasets":[str(x) for x in acquired], "acquired_sha256":acquired_sha256})
        event("finetune_completed", args.pressure, [merged, valid, split], [Path(model), OUT / "model_lineage.json", data / "batch_complete.json"], parent_model=parent, model=model, batch_number=batch_number, batch_epochs=epochs, validation_policy="randomized_30_percent_campaign_only", validation_fraction=fraction, validation_seed=split_seed)


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(func=command_init)
    sub.add_parser("status").set_defaults(func=command_status)
    q=sub.add_parser("inherit-dft", help="Snapshot validated acquisition DFT labels from an earlier run into this named run.")
    q.add_argument("--from-run", default="", help="Source run directory below outputsfull/X_semiactivelearning; omit for the legacy/default run.")
    q.set_defaults(func=command_inherit_dft)
    q=sub.add_parser("reuse-dft", help="Reuse safe unfinished DFT inputs and cluster-selection state from an earlier run.")
    q.add_argument("--pressure", type=float, required=True)
    q.add_argument("--from-run", default="", help="Source run directory below outputsfull/X_semiactivelearning; omit for legacy/default.")
    q.add_argument("--source-kind", default="acquisition_r2")
    q.add_argument("--target-kind", default="acquisition_r3")
    q.set_defaults(func=command_reuse_dft)
    q=sub.add_parser("restart-foundation", help="Begin a new immutable branch from polar-1-s; preserve all prior batches.")
    q.add_argument("--reason", default="Validation regression: restart semi-active fine tuning from polar-1-s")
    q.set_defaults(func=command_restart_foundation)
    q=sub.add_parser("promote-best", help="Pin the best completed registry checkpoint as the next MD parent.")
    q.add_argument("--pressure", type=float, required=True, help="Pressure represented by the completed fine-tuning candidates.")
    q.add_argument("--metric", choices=("validation_loss", "force_rmse", "energy_rmse"), default="validation_loss")
    q.set_defaults(func=command_promote_best)
    for name, func in (("md", command_md), ("analyze", command_analyze), ("approve", command_approve)):
        q=sub.add_parser(name); q.add_argument("--pressure", type=float, required=True); q.set_defaults(func=func)
        if name == "md": q.add_argument("--run", action="store_true")
        if name == "analyze":
            q.add_argument("--trajectory", type=Path, help="Immutable external or campaign trajectory to analyse.")
            q.add_argument("--production-start-frame", type=int, default=0, help="First production frame, zero based; earlier frames are excluded.")
            q.add_argument("--production-frame-stride", type=int, default=1, help="Analyse every Nth production frame; must be positive.")
            q.add_argument("--frame-time-ps", type=float, help="Time between stored trajectory frames, in ps; inferred for campaign trajectories.")
        if name == "approve": q.add_argument("--reason", required=True)
    def kind(value: str) -> str:
        if value == "diagnostic" or re.fullmatch(r"acquisition(?:_r[2-9][0-9]*)?", value): return value
        raise argparse.ArgumentTypeError("kind must be diagnostic, acquisition, or acquisition_r<N> (N >= 2)")
    q=sub.add_parser("clusters"); q.add_argument("--pressure", type=float, required=True); q.add_argument("--kind", type=kind, required=True); q.set_defaults(func=command_clusters)
    q=sub.add_parser("dft"); q.add_argument("--pressure", type=float, required=True); q.add_argument("--kind", type=kind, required=True); q.add_argument("--submit", action="store_true", help="Submit the prepared job to Slurm."); q.add_argument("--run-local", action="store_true", help="Run already-prepared inputs locally, one 24-rank ORCA job at a time; failed jobs are recorded and skipped."); q.add_argument("--replace-prepared", action="store_true", help="Archive unused prepared inputs before regenerating them; refuses if output files exist."); q.add_argument("--collect", action="store_true"); q.add_argument("--parity", action="store_true", help="Re-evaluate MACE parity from already validated DFT frames."); q.set_defaults(func=command_dft)
    q=sub.add_parser("finetune"); q.add_argument("--pressure", type=float, required=True); q.add_argument("--run", action="store_true"); q.add_argument("--epochs", type=int, help="Epochs in this continuation batch; defaults to finetune.batch_epochs."); q.add_argument("--changed-parameter", required=True); q.add_argument("--change-note", required=True); q.set_defaults(func=command_finetune)
    return p


if __name__ == "__main__":
    args=parser().parse_args(); args.func(args)
