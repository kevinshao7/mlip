#!/usr/bin/env python3
"""Deterministic no-agent campaign controller.

Run ``python autonomous.py --once`` repeatedly (for example via a user
systemd timer), or ``python autonomous.py --watch`` in a persistent terminal.
It launches at most one GPU-0 MD task, one GPU-1 MACE task, and one CPU task.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import campaign

ROOT = Path(__file__).resolve().parent
OUT = campaign.OUT
POLICY = ROOT / "autonomous_policy.json"
DIRECTIVES = ROOT / "operator_priorities.txt"
STATE = OUT / "autonomous_state.json"
HEARTBEAT_LOG = OUT / "autonomous_logs" / "controller_heartbeat.log"
_last_dashboard_update = 0.0


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def operator_directives() -> dict[str, object]:
    """Read simple human-editable ``key = value`` scheduling directives.

    This is deliberately deterministic rather than natural-language parsing:
    prose belongs in comments and only documented keys change scheduling.
    """
    values: dict[str, object] = {}
    try:
        lines = DIRECTIVES.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for raw in lines:
        text = raw.split("#", 1)[0].strip()
        if "=" not in text:
            continue
        key, value = (part.strip() for part in text.split("=", 1))
        lower = value.lower()
        if lower in {"true", "false"}:
            values[key] = lower == "true"
        else:
            try:
                values[key] = float(value)
            except ValueError:
                values[key] = value
    return values


def scheduled_pressures(pressures: list[float], directives: dict[str, object]) -> list[float]:
    """Limit work to an operator-selected pressure without touching MD."""
    active = directives.get("active_pressure_gpa", "all")
    if isinstance(active, str) and active.lower() == "all":
        return pressures
    try:
        active_value = float(active)
    except (TypeError, ValueError):
        return pressures
    match = min(pressures, key=lambda p: abs(p - active_value))
    if abs(match - active_value) > 1e-5:
        print(f"[autonomous] directive active_pressure_gpa={active_value} is not a configured pressure; ignoring", flush=True)
        return pressures
    return [match]


def acquisition_kind_for_cycle(cdir: Path, directives: dict[str, object]) -> str:
    """Return the current immutable acquisition round for one pressure."""
    requested = str(directives.get("active_acquisition_kind", "acquisition"))
    if requested != "auto":
        return requested
    names = {"acquisition"}
    for parent in (cdir / "clusters", cdir / "dft"):
        if parent.exists():
            names.update(path.name for path in parent.glob("acquisition*"))
    def number(name: str) -> int:
        match = re.fullmatch(r"acquisition(?:_r(\d+))?", name)
        return int(match.group(1) or 1) if match else -1
    ordered = sorted((name for name in names if number(name) > 0), key=number)
    def has_runnable_orca_input(name: str) -> bool:
        """True only while a prepared round has work that is not terminal.

        Completed and intentionally skipped SCF jobs are terminal.  Treating
        their manifest as permanently runnable caused ``auto`` to select the
        same completed round forever, leaving continuous acquisition unable to
        advance to a new non-duplicate cluster selection.
        """
        ddir = cdir / "dft" / name
        manifest = ddir / "manifest.csv"
        if not manifest.is_file():
            return False
        try:
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            failed = load_json(ddir / "dft_failure_report.json", {}).get("failures", [])
            terminal_skips = {str(row.get("stem")) for row in failed
                              if row.get("action") == "skipped_no_scf_retry"}
        except (OSError, csv.Error):
            # Preserve an unreadable manifest for operator diagnosis instead
            # of silently advancing past it.
            return True
        for row in rows:
            stem = str(row.get("stem", ""))
            output = ddir / "outputs" / f"{stem}.out"
            if not campaign.local_orca_complete(output) and stem not in terminal_skips:
                return True
        return False

    # A prepared ORCA manifest with actual runnable work is always first
    # priority. This is crucial for named/restart loops, but a fully terminal
    # round must no longer block r(N+1).
    prepared = [name for name in ordered
                if (cdir / "dft" / name / "manifest.csv").exists()
                and has_runnable_orca_input(name)]
    if prepared:
        return prepared[0]
    # Next resume a fully selected but not-yet-prepared acquisition round.
    selected = [name for name in ordered
                if (cdir / "clusters" / name / "clusters.extxyz").exists()
                and not (cdir / "dft" / name / "manifest.csv").exists()]
    if selected:
        return selected[0]
    if directives.get("continuous_acquisition", False):
        return f"acquisition_r{number(ordered[-1]) + 1}"
    return ordered[-1]


def matching_task_process(task: dict) -> bool:
    """Detect a worker after its short-lived launch-shell PID has changed.

    The controller only treats a matching process as live when its command
    line names this exact pressure-cycle directory.  This prevents a watcher
    restart from launching a duplicate MD/ORCA worker.
    """
    cycle_dir = campaign.cycle(float(task["pressure_gpa"]))
    dft_dir = str(cycle_dir / "dft" / "acquisition")
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        # The MD launcher is executed with its cycle directory as cwd, so its
        # command line can be only ``python run_md.py``.
        if task["role"] == "gpu0" and "run_md.py" in cmdline:
            try:
                if os.readlink(proc / "cwd") == str(cycle_dir / "md"):
                    return True
            except OSError:
                pass
        # ORCA receives an absolute input path under this exact DFT directory.
        if task["role"] == "cpu" and dft_dir in cmdline and ("orca" in cmdline.lower() or "campaign.py" in cmdline):
            return True
    return False


def active(task: dict) -> bool:
    exit_file = Path(task["exit_file"])
    if exit_file.exists():
        return False
    try:
        os.kill(int(task["pid"]), 0)
        return True
    except OSError:
        return matching_task_process(task)


def task_for_role(state: dict, role: str) -> bool:
    return any(task["role"] == role and active(task) for task in state["tasks"].values())


def progress_snapshot(task: dict) -> str:
    """Return the newest compact MD/DFT progress token from a worker log."""
    try:
        with Path(task["log"]).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 8192))
            recent = stream.read().decode(errors="replace").replace("\r", "\n")
    except OSError:
        return "progress=log-unavailable"
    if task["role"] == "gpu0":
        matches = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[([^\]]*)\]", recent)
        if matches:
            step, total, timing = matches[-1]
            return f"md_steps={step}/{total};eta={timing.split('<')[-1] if '<' in timing else timing}"
    if task["role"] == "cpu":
        matches = re.findall(r"\[local-dft\]\s+(\d+)/(\d+)\s+([^\n]+)", recent)
        if matches:
            index, total, detail = matches[-1]
            return f"dft={index}/{total};{detail[:90]}"
    return "progress=awaiting-log-update"


def heartbeat(state: dict, policy: dict | None = None) -> None:
    """One compact, grep-friendly scheduler status line per controller poll."""
    by_role = {task["role"]: task for task in state["tasks"].values() if active(task)}
    states = []
    for role in ("gpu0", "gpu1", "cpu"):
        task = by_role.get(role)
        if task is None:
            states.append(f"{role}=idle")
        else:
            started = task["started_utc"].replace("+00:00", "Z")
            states.append(f"{role}=running:{task['key']}:pid={task['pid']}:since={started}:{progress_snapshot(task)}:log={task['log']}")
    if policy is not None and not policy.get("controller_may_launch_md", False):
        next_md = "manual (controller MD disabled)"
    else:
        cfg = campaign.config()
        unlaunched = [campaign.tag(float(p)) for p in cfg["pressures_gpa"] if not list((campaign.cycle(float(p)) / "md").glob("production_*.xyz"))]
        next_md = unlaunched[0] if unlaunched else "none"
    directives = operator_directives()
    focus = directives.get("active_pressure_gpa", "all")
    line = f"[autonomous status] utc={utc()} {' | '.join(states)} | focus={focus}GPa | next_md={next_md}"
    print(line, flush=True)
    HEARTBEAT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HEARTBEAT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def maybe_update_dashboard(policy: dict) -> None:
    """Refresh the read-only campaign figure at a bounded cadence."""
    global _last_dashboard_update
    interval = float(policy.get("dashboard_update_seconds", 30))
    if interval <= 0 or time.monotonic() - _last_dashboard_update < interval:
        return
    try:
        from campaign_dashboard import render
        render()
        _last_dashboard_update = time.monotonic()
    except Exception as error:  # a dashboard failure must never interrupt jobs
        print(f"[autonomous] dashboard update failed: {error}", flush=True)


def command_text(command: list[str]) -> str:
    import shlex
    return shlex.join(command)


def launch(state: dict, key: str, role: str, pressure: float, command: list[str], env_extra: dict[str, str] | None = None, dry_run: bool = False, cwd: Path = ROOT) -> None:
    if key in state["tasks"] and active(state["tasks"][key]):
        return
    if dry_run:
        print(f"[autonomous dry-run] would start {key} ({role}): {command_text(command)}", flush=True)
        return
    logs = OUT / "autonomous_logs"; logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{key}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    exit_file = log.with_suffix(".exitcode")
    environment = os.environ.copy(); environment.update(env_extra or {})
    wrapped = f"{command_text(command)}; status=$?; echo $status > {command_text([str(exit_file)])}; exit $status"
    stream = log.open("w", encoding="utf-8")
    process = subprocess.Popen(["/bin/bash", "-lc", wrapped], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                               env=environment, start_new_session=True)
    stream.close()
    state["tasks"][key] = {"key":key, "role":role, "pressure_gpa":pressure, "pid":process.pid,
                           "command":command, "log":str(log), "exit_file":str(exit_file), "started_utc":utc()}
    campaign.event("autonomous_task_started", pressure, [POLICY], [], task_key=key, role=role, command=command)
    print(f"[autonomous] started {key} ({role}), pid={process.pid}; log={log}", flush=True)


def poll(state: dict) -> None:
    for key, task in list(state["tasks"].items()):
        exit_file = Path(task["exit_file"])
        if not exit_file.exists() and active(task):
            continue
        code_text = exit_file.read_text(encoding="utf-8").strip() if exit_file.exists() else "-998"
        code = int(code_text) if code_text.lstrip("-").isdigit() else -999
        task["finished_utc"] = utc(); task["exit_code"] = code
        state["completed_tasks"].append(task)
        del state["tasks"][key]
        campaign.event("autonomous_task_finished", task["pressure_gpa"], [exit_file], [], task_key=key,
                       role=task["role"], exit_code=code, log=task["log"])
        if key.startswith("finetune_") and code == 0:
            cdir = campaign.cycle(float(task["pressure_gpa"]))
            save_json(cdir / "training" / "autonomous_finetune_complete.json",
                      {"utc":utc(), "task":key, "model":campaign.current_parent(), "log":task["log"]})
        print(f"[autonomous] finished {key}, exit={code}", flush=True)


def decision(cdir: Path, name: str, value: dict, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[autonomous dry-run] decision {cdir.name}/{name}: {value}", flush=True)
        return
    path = cdir / "autonomous_decisions.json"
    data = load_json(path, {"policy": str(POLICY), "decisions": {}})
    data["decisions"][name] = {"utc":utc(), **value}
    save_json(path, data)


def command(*items: object) -> list[str]:
    return [sys.executable, str(ROOT / "campaign.py"), *map(str, items)]


def has_uncollected_completed_dft(ddir: Path) -> bool:
    """Return true when a finished ORCA output is newer than the DFT snapshot.

    ORCA writes ``.out.partial`` while a calculation is in progress and renames
    it to ``.out`` only after it exits.  Thus a collection can safely run on
    GPU 1 while the CPU continues the serial DFT queue; it never reads a file
    still being written.
    """
    processed = ddir / "processed.extxyz"
    completed = list((ddir / "outputs").glob("*.out")) if (ddir / "outputs").is_dir() else []
    return bool(completed) and (not processed.exists() or max(path.stat().st_mtime for path in completed) > processed.stat().st_mtime)


def reconcile(policy: dict, state: dict, dry_run: bool) -> None:
    cfg = campaign.config()
    pressures = [float(p) for p in cfg["pressures_gpa"]]
    directives = operator_directives()
    work_pressures = scheduled_pressures(pressures, directives)
    # MD production is user-owned. The controller may observe existing
    # trajectories but does not launch or terminate MD by default.
    if policy.get("controller_may_launch_md", False) and not task_for_role(state, "gpu0"):
        for pressure in work_pressures:
            md = campaign.cycle(pressure) / "md"
            if not list(md.glob("production_*.xyz")):
                if (md / "manifest.json").exists() and (md / "run_md.sh").exists():
                    md_command, md_cwd = [str(md / "run_md.sh")], md
                else:
                    md_command, md_cwd = command("md", "--pressure", pressure, "--run"), ROOT
                launch(state, f"md_{campaign.tag(pressure)}", "gpu0", pressure, md_command,
                       {"CUDA_VISIBLE_DEVICES": policy["md_gpu"]}, dry_run, md_cwd)
                break

    # At most one CPU workflow action.  DFT remains CPU-only and is launched
    # after selection; failed ORCA structures are preserved/skipped by campaign.
    cpu_pressures = [] if task_for_role(state, "cpu") else work_pressures
    for pressure in cpu_pressures:
        cdir = campaign.cycle(pressure); md = cdir / "md"; analysis = cdir / "analysis"
        acquisition_kind = acquisition_kind_for_cycle(cdir, directives)
        trajectories = sorted(md.glob("production_*.xyz"))
        ddir = cdir / "dft" / acquisition_kind
        # Resume an existing queue before looking for a local MD.  Reused DFT
        # queues retain the external production trajectory in analysis summary
        # provenance and must not require a duplicate MD file in this run.
        if (ddir / "manifest.csv").exists() and not (ddir / "dft_failure_report.json").exists():
            launch(state, f"dft_{campaign.tag(pressure)}", "cpu", pressure,
                   command("dft", "--pressure", pressure, "--kind", acquisition_kind, "--run-local"), dry_run=dry_run)
            break
        if not (analysis / "summary.json").exists():
            if not trajectories:
                continue
            launch(state, f"analyze_{campaign.tag(pressure)}", "cpu", pressure, command("analyze", "--pressure", pressure), dry_run=dry_run)
            break
        summary = load_json(analysis / "summary.json", {})
        decision(cdir, "equilibration", {"blocks":False, "mean_pressure_gpa":summary.get("mean_pressure_GPa"),
                 "pressure_std_gpa":summary.get("pressure_std_GPa"), "volume_start_A3":summary.get("volume_start_A3"),
                 "volume_end_A3":summary.get("volume_end_A3")}, dry_run)
        if not summary.get("cluster_eligible", False):
            decision(cdir, "cluster_selection", {"action":"skipped", "reason":"species gate not met"}, dry_run)
            continue
        clusters = cdir / "clusters" / acquisition_kind / "clusters.extxyz"
        approval = cdir / "approval_acquisition.json"
        if not approval.exists():
            launch(state, f"approve_{campaign.tag(pressure)}", "cpu", pressure,
                   command("approve", "--pressure", pressure, "--reason", "deterministic autonomous policy: illegal fraction below 5%; acquisition permitted"), dry_run=dry_run)
            break
        if not clusters.exists():
            launch(state, f"clusters_{campaign.tag(pressure)}", "cpu", pressure,
                   command("clusters", "--pressure", pressure, "--kind", acquisition_kind), dry_run=dry_run)
            break
        if not (ddir / "manifest.csv").exists():
            launch(state, f"dft_prepare_{campaign.tag(pressure)}", "cpu", pressure,
                   command("dft", "--pressure", pressure, "--kind", acquisition_kind), dry_run=dry_run)
            break
        if not (ddir / "dft_failure_report.json").exists():
            launch(state, f"dft_{campaign.tag(pressure)}", "cpu", pressure,
                   command("dft", "--pressure", pressure, "--kind", acquisition_kind, "--run-local"), dry_run=dry_run)
            break

    # GPU 1: parity collection then conditional fine tuning. The collect command
    # inherits CUDA_VISIBLE_DEVICES=1 so MACE parity never competes with MD.
    if task_for_role(state, "gpu1"):
        return
    for pressure in work_pressures:
        cdir = campaign.cycle(pressure); acquisition_kind = acquisition_kind_for_cycle(cdir, directives); ddir = cdir / "dft" / acquisition_kind
        # While the newest round is waiting for its first ORCA result, keep
        # GPU 1 training/parity-checking the latest prior validated round.
        # As soon as the new round has a completed output it takes priority.
        if not (ddir / "processed.extxyz").exists() and not has_uncollected_completed_dft(ddir):
            prior = sorted((cdir / "dft").glob("acquisition*/processed.extxyz"), key=lambda path: path.stat().st_mtime, reverse=True)
            if prior:
                ddir = prior[0].parent; acquisition_kind = ddir.name
        # Do not wait for all requested DFT jobs.  Snapshot only completed
        # ``.out`` files while the CPU queue continues, then parity-check and
        # fine tune a 20-epoch batch from the validated snapshot.  This keeps
        # GPU 1 useful and preserves each collection/training snapshot in its
        # respective history directory.
        # While a CPU DFT queue is still active, a previously collected
        # snapshot remains eligible for parity/fine tuning even during the
        # interval before the next ORCA result arrives.  Only skip a live
        # queue when it has neither a collected snapshot nor a new completed
        # output to collect.
        if (not has_uncollected_completed_dft(ddir)
                and not (ddir / "processed.extxyz").exists()
                and not (ddir / "dft_failure_report.json").exists()):
            continue
        processed = ddir / "processed.extxyz"
        # Validation is mandatory independently of the 30-label training
        # minimum.  A missing parity artifact means a collection must be
        # (re)run, including when only one successful ORCA label exists.
        if has_uncollected_completed_dft(ddir) or not processed.exists() or not (ddir / "parity_summary.json").exists():
            launch(state, f"collect_{campaign.tag(pressure)}", "gpu1", pressure,
                   command("dft", "--pressure", pressure, "--kind", acquisition_kind, "--collect"),
                   {"CUDA_VISIBLE_DEVICES": policy["mace_gpu"]}, dry_run)
            return
        parity = load_json(ddir / "parity_summary.json", {})
        # A 20-epoch batch changes the parent model. Re-evaluate its parity
        # before deciding on the next batch, using all currently collected
        # DFT data for this pressure.
        if parity.get("parent_model") != campaign.current_parent():
            launch(state, f"parity_{campaign.tag(pressure)}", "gpu1", pressure,
                   command("dft", "--pressure", pressure, "--kind", acquisition_kind, "--parity"),
                   {"CUDA_VISIBLE_DEVICES": policy["mace_gpu"]}, dry_run)
            return
        if (ddir / "collection.csv").exists():
            with (ddir / "collection.csv").open(encoding="utf-8") as f:
                accepted = sum(1 for row in csv.DictReader(f) if row.get("status") == "accepted")
        else:
            accepted = 0
        mae = parity.get("energy_mae_mev_per_atom")
        lineage = load_json(OUT / "model_lineage.json", {})
        prior_batches = [entry for entry in lineage.get("history", []) if entry.get("finetune_id", "").startswith("sal_")]
        bootstrap = (policy.get("bootstrap_finetune_with_any_validated_campaign_dft", False)
                     and not prior_batches and accepted >= policy["minimum_successful_dft_clusters_for_finetune"])
        eligible = bootstrap or (accepted >= policy["minimum_successful_dft_clusters_for_finetune"] and isinstance(mae, (int, float)) and mae > policy["parity_energy_mae_trigger_mev_per_atom"])
        # Optionally bind one batch to each immutable DFT snapshot. The
        # operator can disable this to keep GPU 1 continuously fine tuning.
        snapshot_sha = campaign.sha(processed)
        source_key = str(processed.resolve())
        if directives.get("finetune_on_new_dft_only", False):
            already_trained = any(entry.get("pressure_gpa") == pressure and entry.get("acquired_sha256", {}).get(source_key) == snapshot_sha for entry in prior_batches)
            eligible = eligible and not already_trained
        if directives.get("continue_finetune", True) is False:
            eligible = False
        decision(cdir, "fine_tune_gate", {"accepted_clusters":accepted, "energy_mae_mev_per_atom":mae,
                 "minimum_clusters":policy["minimum_successful_dft_clusters_for_finetune"],
                 "mae_trigger":policy["parity_energy_mae_trigger_mev_per_atom"], "bootstrap":bootstrap, "eligible":eligible,
                 "snapshot_sha256":snapshot_sha, "operator_directives":directives}, dry_run)
        if not eligible:
            continue
        launch(state, f"finetune_{campaign.tag(pressure)}", "gpu1", pressure,
               command("finetune", "--pressure", pressure, "--run", "--epochs", policy["finetune_batch_epochs"], "--changed-parameter", "autonomous_dft_acquisition",
                       "--change-note", "20-epoch continuation batch; parity and new DFT data rechecked before another batch"),
               {"CUDA_VISIBLE_DEVICES": policy["mace_gpu"]}, dry_run)
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--status", action="store_true", help="Print the persisted scheduler state without launching work.")
    parser.add_argument("--interval", type=int, help="Override policy polling interval for --watch.")
    parser.add_argument("--dry-run", action="store_true", help="Print the next launch decisions without changing campaign state.")
    args = parser.parse_args()
    policy = load_json(POLICY, {})
    required = {"minimum_successful_dft_clusters_for_finetune", "parity_energy_mae_trigger_mev_per_atom", "md_gpu", "mace_gpu", "finetune_batch_epochs", "bootstrap_finetune_with_any_validated_campaign_dft", "controller_may_launch_md", "dashboard_update_seconds"}
    missing = required - policy.keys()
    if missing: raise ValueError(f"Policy missing required keys: {sorted(missing)}")
    if args.status:
        state = load_json(STATE, {"policy":str(POLICY), "tasks":{}, "completed_tasks":[]})
        heartbeat(state, policy)
        return
    while True:
        state = load_json(STATE, {"policy":str(POLICY), "tasks":{}, "completed_tasks":[]})
        if not args.dry_run: poll(state)
        reconcile(policy, state, args.dry_run)
        heartbeat(state, policy)
        if not args.dry_run:
            maybe_update_dashboard(policy)
        if not args.dry_run: save_json(STATE, state)
        if args.once: return
        time.sleep(args.interval or int(policy["poll_interval_seconds"]))


if __name__ == "__main__":
    main()
