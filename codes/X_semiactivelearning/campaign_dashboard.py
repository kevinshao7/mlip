#!/usr/bin/env python3
"""Render one self-updating visual summary of the semi-active-learning loop.

The dashboard is read-only: it derives all state from the campaign output
root, so it is safe to run while MD, ORCA, or MACE jobs are active.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import campaign

ROOT = Path(__file__).resolve().parent
OUT = campaign.OUT
POLICY = ROOT / "autonomous_policy.json"
TARGET = OUT / "campaign_dashboard.png"
METADATA = OUT / "campaign_dashboard.json"
PUBLIC_TARGET = campaign.BASE_OUT / "campaign_dashboard.png"
ACTIVE_POINTER = campaign.BASE_OUT / "active_dashboard.json"


def load(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def operator_note() -> str:
    path = ROOT / "operator_priorities.txt"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("note") and "=" in line:
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    return "No operator priority note"


def count_rows(path: Path) -> tuple[int, int]:
    """Return requested and accepted collection counts."""
    if not path.exists():
        return 0, 0
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return len(rows), sum(row.get("status") == "accepted" for row in rows)


def rows() -> list[dict]:
    result = []
    for pressure in map(float, campaign.config()["pressures_gpa"]):
        cycle = campaign.cycle(pressure)
        md = cycle / "md"; analysis = cycle / "analysis"; dfts = sorted((cycle / "dft").glob("acquisition*"))
        requested = accepted = partial = 0; parity_candidates = []
        for dft in dfts:
            requested_one, accepted_one = count_rows(dft / "collection.csv")
            if not requested_one and (dft / "manifest.csv").exists():
                with (dft / "manifest.csv").open(encoding="utf-8") as stream:
                    requested_one = sum(1 for _ in csv.DictReader(stream))
            requested += requested_one; accepted += accepted_one
            partial += len(list((dft / "outputs").glob("*.partial"))) if (dft / "outputs").exists() else 0
            if (dft / "parity_summary.json").exists(): parity_candidates.append(dft / "parity_summary.json")
        parity = load(max(parity_candidates, key=lambda p: p.stat().st_mtime), {}) if parity_candidates else {}
        summary = load(analysis / "summary.json", {})
        result.append({
            "pressure": pressure, "tag": campaign.tag(pressure),
            "md": bool(list(md.glob("production_*.xyz"))),
            "analysis": (analysis / "summary.json").exists(),
            "clusters": (cycle / "clusters" / "acquisition" / "clusters.extxyz").exists(),
            "prepared": bool(dfts),
            "dft_done": any((dft / "dft_failure_report.json").exists() for dft in dfts),
            "requested": requested, "accepted": accepted, "partial": partial,
            "mae": parity.get("energy_mae_mev_per_atom"),
            "mean_pressure": summary.get("mean_pressure_GPa"),
        })
    return result


METRIC_RE = re.compile(
    r"(?:Initial|Epoch\s+(?P<epoch>\d+)):\s+head:.*?loss=\s*(?P<loss>[0-9.eE+-]+),"
    r"\s*RMSE_E_per_atom=\s*(?P<energy>[0-9.eE+-]+)\s*meV,"
    r"\s*RMSE_F=\s*(?P<force>[0-9.eE+-]+)\s*meV\s*/\s*A"
)
TABLE_ROW_RE = re.compile(
    r"\|\s*(?P<split>train|valid)_[^|]*\|\s*"
    r"(?P<energy>[0-9.eE+-]+)\s*\|\s*(?P<force>[0-9.eE+-]+)\s*\|",
    re.IGNORECASE,
)


def batch_metrics(history: list[dict]) -> list[dict]:
    """Read validation trajectories and saved-model train/validation RMSEs."""
    batches = []
    for item in history:
        model = Path(item.get("model", ""))
        run_dir = model.parents[1] if len(model.parents) > 1 else Path()
        logs = sorted((run_dir / "logs").glob("*.log")) if run_dir.exists() else []
        text = logs[-1].read_text(errors="replace") if logs else ""
        series = []
        for line in text.splitlines():
            hit = METRIC_RE.search(line)
            if hit:
                series.append({"epoch": -1 if hit.group("epoch") is None else int(hit.group("epoch")),
                               "loss": float(hit.group("loss")), "energy": float(hit.group("energy")),
                               "force": float(hit.group("force"))})
        # MACE writes one table for the selected stage-one model and another
        # for the saved stage-two/SWA model.  The final row of each split is
        # therefore the endpoint actually handed to the next lineage batch.
        endpoint = {}
        for hit in TABLE_ROW_RE.finditer(text):
            endpoint[hit.group("split").lower()] = {
                "energy": float(hit.group("energy")), "force": float(hit.group("force"))
            }
        if series or endpoint:
            batches.append({"item": item, "series": series, "endpoint": endpoint})
    return batches


def publish_active_named_run(output: Path) -> None:
    """Publish a clearly tracked top-level view of the active named run.

    The historical unnamed dashboard used the same top-level filename.  Archive
    that image before replacing it with a copy of the named-run dashboard so a
    researcher opening the conventional path never mistakes old lineage for
    the active loop.
    """
    if not campaign.RUN_ID or output.resolve() != TARGET.resolve():
        return
    previous = load(ACTIVE_POINTER, {})
    if PUBLIC_TARGET.exists() and previous.get("run_id") != campaign.RUN_ID:
        archive = campaign.BASE_OUT / "dashboard_history"
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(PUBLIC_TARGET, archive / f"campaign_dashboard_{previous.get('run_id', 'unnamed')}_{stamp}.png")
    temporary = PUBLIC_TARGET.with_suffix(".active.partial.png")
    shutil.copy2(output, temporary)
    temporary.replace(PUBLIC_TARGET)
    ACTIVE_POINTER.write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(),
                                          "run_id": campaign.RUN_ID,
                                          "source_dashboard": str(output),
                                          "public_dashboard": str(PUBLIC_TARGET)}, indent=2)+"\n", encoding="utf-8")


def render(output: Path = TARGET) -> None:
    policy = load(POLICY, {})
    data = rows(); lineage = load(OUT / "model_lineage.json", {"history": []})
    threshold = policy.get("parity_energy_mae_trigger_mev_per_atom", 30.0)
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.1, 1, 1))
    ax_stage = fig.add_subplot(grid[0, 0]); ax_dft = fig.add_subplot(grid[0, 1])
    ax_parity = fig.add_subplot(grid[1, 0]); ax_loss = fig.add_subplot(grid[1, 1])
    ax_error = fig.add_subplot(grid[2, 0]); ax_model = fig.add_subplot(grid[2, 1])

    labels = [f"{row['pressure']:.2f}" for row in data]; y = np.arange(len(data))
    stages = [("MD", "md", "#2878b5"), ("analysis", "analysis", "#59a14f"),
              ("allowed clusters", "clusters", "#edc948"), ("DFT prepared", "prepared", "#f28e2b"),
              ("DFT queue complete", "dft_done", "#e15759")]
    for xpos, (label, key, color) in enumerate(stages):
        present = [row[key] for row in data]
        ax_stage.scatter(np.full(sum(present), xpos), y[np.array(present, dtype=bool)], s=105,
                         c=color, marker="s", label=label, edgecolors="white", linewidths=.7)
    ax_stage.set(yticks=y, yticklabels=[f"{p} GPa" for p in labels], xlim=(-.6, len(stages)-.4),
                 xticks=range(len(stages)), xticklabels=[x[0] for x in stages], title="Pressure-cycle stage map")
    ax_stage.tick_params(axis="x", rotation=25); ax_stage.grid(axis="y", alpha=.25); ax_stage.legend(fontsize=8, loc="lower right")

    requested = np.array([row["requested"] for row in data]); accepted = np.array([row["accepted"] for row in data])
    partial = np.array([row["partial"] for row in data]); pending = np.maximum(requested - accepted - partial, 0)
    ax_dft.bar(y, accepted, color="#59a14f", label="validated")
    ax_dft.bar(y, partial, bottom=accepted, color="#e15759", label="SCF skipped")
    ax_dft.bar(y, pending, bottom=accepted+partial, color="#bab0ac", label="not yet classified")
    ax_dft.set(xticks=y, xticklabels=labels, xlabel="target pressure / GPa", ylabel="clusters", title="DFT acquisition status")
    ax_dft.legend(fontsize=8); ax_dft.grid(axis="y", alpha=.25)

    p = np.array([row["pressure"] for row in data]); mae = np.array([np.nan if row["mae"] is None else row["mae"] for row in data], float)
    ax_parity.semilogx(p, mae, "o-", color="#4e79a7", label="energy MAE")
    ax_parity.axhline(threshold, color="#e15759", ls="--", label=f"fine-tune trigger ({threshold:g} meV/atom)")
    for xx, yy, n in zip(p, mae, accepted):
        if np.isfinite(yy): ax_parity.annotate(f"n={n}", (xx, yy), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax_parity.set(xlabel="target pressure / GPa", ylabel="meV/atom", title="Parity diagnostic on validated campaign DFT")
    ax_parity.grid(alpha=.25); ax_parity.legend(fontsize=8)

    active_branch = str(lineage.get("active_branch", "legacy"))
    # The dashboard's error trends must never splice a deliberately abandoned
    # lineage into a fresh foundation restart.  The complete history remains
    # immutable in model_lineage.json and its branch snapshots.
    history = [item for item in lineage.get("history", [])
               if str(item.get("branch_id", "legacy")) == active_branch]
    metric_batches = batch_metrics(history)
    if metric_batches:
        latest = metric_batches[-1]
        epochs = [row["epoch"] for row in latest["series"]]
        ax_loss.plot(epochs, [row["loss"] for row in latest["series"]], "o-", ms=3, color="#f28e2b")
        ax_loss.set(xlabel="epoch (−1 = initial)", ylabel="weighted validation loss",
                    title=f"Latest fine-tuning batch loss: {latest['item'].get('finetune_id', 'unknown')}")
        ax_loss.grid(alpha=.25)
        endpoints = [batch["endpoint"] for batch in metric_batches]
        batch_ids = np.arange(1, len(endpoints)+1)
        # Explicit train/validation separation makes divergence (overfitting
        # or distribution shift) visible rather than hiding it in one number.
        for split, color, style, label in (
            ("train", "#4e79a7", "--", "train energy RMSE / atom"),
            ("valid", "#4e79a7", "-", "validation energy RMSE / atom"),
            ("train", "#e15759", "--", "train force RMSE"),
            ("valid", "#e15759", "-", "validation force RMSE"),
        ):
            quantity = "energy" if "energy" in label else "force"
            values = [row.get(split, {}).get(quantity, np.nan) for row in endpoints]
            marker = "o" if quantity == "energy" else "s"
            ax_error.plot(batch_ids, values, marker=marker, ls=style, label=label, color=color, ms=4)
        ax_error.set(xlabel="completed fine-tuning batch", ylabel="RMSE / meV", title="Saved-model endpoint RMSE (train vs validation)")
        ax_error.grid(alpha=.25); ax_error.legend(fontsize=8)
    else:
        for axis, title in ((ax_loss, "Fine-tuning loss"), (ax_error, "Fine-tuning errors")):
            axis.text(.5, .5, "No completed MACE metric log", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(title)
    if history:
        dates = [datetime.fromisoformat(x["utc"]) for x in history]
        pressures = [x.get("pressure_gpa", np.nan) for x in history]
        ax_model.plot(dates, pressures, "o-", color="#af7aa1")
        for i, item in enumerate(history): ax_model.annotate(f"B{item.get('batch_number', i+1)}", (dates[i], pressures[i]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax_model.set_ylabel("acquisition pressure / GPa")
    else:
        ax_model.text(.5, .5, "No completed fine-tuning batch", ha="center", va="center", transform=ax_model.transAxes)
    ax_model.set_title("Fine-tuning lineage (each point = immutable 20-epoch batch)")
    ax_model.grid(alpha=.25)
    for label in ax_model.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
    current = Path(lineage.get("current_model", "polar-1-s")).name
    run_label = campaign.RUN_ID or "unnamed legacy run"
    fig.suptitle(f"Semi-active-learning campaign dashboard [{run_label}] — updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\nactive branch: {active_branch} | current model: {current}\npriority: {operator_note()}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.png")
    fig.savefig(temporary, dpi=170, bbox_inches="tight")
    temporary.replace(output); plt.close(fig)
    METADATA.write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), "dashboard": str(output), "current_model": lineage.get("current_model"), "active_branch": active_branch, "fine_tune_trigger_mev_per_atom": threshold, "operator_priority_note": operator_note()}, indent=2)+"\n", encoding="utf-8")
    publish_active_named_run(output)
    print(f"Updated {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Refresh the same PNG repeatedly.")
    parser.add_argument("--interval", type=float, default=30, help="Seconds between refreshes in --watch mode.")
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args()
    if args.interval <= 0: parser.error("--interval must be positive")
    while True:
        render(args.output)
        if not args.watch: break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
