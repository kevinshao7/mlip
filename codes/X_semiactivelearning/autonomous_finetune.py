#!/usr/bin/env python3
"""Run two-child, two-GPU 20-epoch MACE fine-tuning tournaments.

Each round starts both children from the same incumbent and immutable data
split. GPU 0 changes only the learning rate to 2x the incumbent rate; GPU 1
changes only it to 0.5x. The saved-model validation force RMSE chooses the
winner after both children finish.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# This controller targets the base campaign (no SAL_RUN_ID).  Explicitly
# clear the variable so that an inherited shell setting cannot redirect
# outputs to a named sub-run like cycle2.
os.environ["SAL_RUN_ID"] = ""

import campaign
from launch_single_gpu import SAVED_METRIC

ROOT = Path(__file__).resolve().parent
MLIP = ROOT.parents[1]
sys.path.insert(0, str(MLIP / "codes"))
RUN_ID = ""
PRESSURE_GPA = 31.6227766017
EPOCHS = 20
D2 = (ROOT / "../D2_naive").resolve()
LAUNCHER = D2 / "launch_single_gpu.py"
EXPERIMENT_LOG = D2 / "hyperparameters.txt"
TRAINING = campaign.cycle(PRESSURE_GPA) / "training"
STATE = campaign.OUT / "autonomous_finetune.json"
LOCK = campaign.OUT / "autonomous_finetune.lock"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_experiment_note(lines: list[str]) -> None:
    """Append one indivisible controller decision to the shared lab notebook."""
    block = "\n" + "\n".join(lines) + "\n"
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(block)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Inspect the resolved first round without writing or launching.")
    parser.add_argument("--max-rounds", type=int, default=0, help="Stop after this many rounds; 0 means continue until neither child improves.")
    parser.add_argument("--interval", type=positive_float, default=30.0, help="Dashboard refresh/poll interval in seconds.")
    args = parser.parse_args()
    if args.max_rounds < 0:
        parser.error("--max-rounds cannot be negative")
    return args


def child_specs(round_number: int, parent: Path, learning_rate: float) -> list[dict]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    specs = []
    for gpu, factor, label in (("0", 2.0, "2x"), ("1", 0.5, "0p5x")):
        child_lr = learning_rate * factor
        child_id = f"sal_P31p6228GPa_autoround{round_number:03d}_gpu{gpu}_{label}_{stamp}"
        note = (
            f"Autonomous sibling round {round_number}: start from the same incumbent and immutable "
            f"P31 split; change only learning rate from {learning_rate:g} to {child_lr:g} ({factor:g}x)."
        )
        command = [
            sys.executable, str(LAUNCHER), "--gpu", gpu, "--",
            f"--foundation-model={parent}",
            f"--train-file={TRAINING / 'cumulative_train.extxyz'}",
            f"--valid-file={TRAINING / 'random_validation.extxyz'}",
            f"--name={child_id}", f"--max-num-epochs={EPOCHS}",
            f"--lr={child_lr}", "--e0s=d4rev",
            f"--reference-method={campaign.config()['dft']['functional_basis']}",
            "--no-swa",
            "--changed-parameter=learning_rate", f"--change-note={note}",
        ]
        specs.append({"gpu": gpu, "factor": factor, "label": label, "lr": child_lr,
                      "finetune_id": child_id, "command": command})
    return specs


def verify_frozen_data(expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = TRAINING / name
        if not path.is_file() or campaign.sha(path) != digest:
            raise RuntimeError(f"Frozen autonomous fine-tuning input changed: {path}")


def prepare_data(parent: Path, learning_rate: float) -> dict[str, str]:
    """Use campaign.py to create and provenance-record the deterministic split."""
    command = [
        sys.executable, str(ROOT / "campaign.py"), "finetune", "--pressure", str(PRESSURE_GPA),
        "--parent-model", str(parent), "--lr", str(learning_rate),
        "--changed-parameter=autonomous_lr_tournament_controller",
        "--change-note=Prepare one immutable P31 split for repeated two-child learning-rate-only tournaments; no optimization is started by this preparation.",
    ]
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "SAL_RUN_ID": RUN_ID})
    files = ("cumulative_train.extxyz", "random_validation.extxyz", "random_split.json")
    hashes = {name: campaign.sha(TRAINING / name) for name in files}
    (TRAINING / "launch_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join([sys.executable, str(Path(__file__).resolve())]) + "\n",
        encoding="utf-8",
    )
    return hashes


def refresh_dashboard() -> None:
    try:
        from campaign_dashboard import render
        render()
    except Exception as exc:
        print(f"[autonomous-finetune] dashboard update failed: {type(exc).__name__}: {exc}", flush=True)


def write_status(state: str, incumbent: dict, round_number: int, children: list[dict], **extra: object) -> None:
    split = json.loads((TRAINING / "random_split.json").read_text(encoding="utf-8"))
    public_children = [{key: value for key, value in child.items() if key != "command"} for child in children]
    status = {
        "state": state, "updated_utc": utc(), "pressure_gpa": PRESSURE_GPA,
        "finetune_id": f"autonomous_round_{round_number:03d}", "batch_number": round_number,
        "batch_epochs": EPOCHS, "parent_model": str(incumbent["model"]),
        "learning_rate": incumbent["learning_rate"], "train_count": split["train_count"],
        "validation_count": split["validation_count"], "children": public_children,
        "changed_parameter": "learning_rate", **extra,
    }
    campaign.write_json(TRAINING / "finetune_status.json", status)
    refresh_dashboard()


def saved_model_result(spec: dict) -> dict:
    run_dirs = sorted((D2 / "runs").glob(f"{spec['finetune_id']}_*"), key=lambda path: path.stat().st_mtime)
    if len(run_dirs) != 1:
        raise RuntimeError(f"Expected one immutable run for {spec['finetune_id']}; found {len(run_dirs)}")
    run_dir = run_dirs[0]
    logs = sorted((run_dir / "logs").glob("*.log"))
    if not logs:
        raise RuntimeError(f"No training log found in {run_dir}")
    endpoint = {}
    for match in SAVED_METRIC.finditer(logs[-1].read_text(errors="replace")):
        endpoint[match.group("split").lower()] = {
            "energy_rmse_mev_per_atom": float(match.group("energy")),
            "force_rmse_mev_per_a": float(match.group("force")),
        }
    if "valid" not in endpoint:
        raise RuntimeError(f"No saved-model validation endpoint found in {logs[-1]}")
    model = run_dir / "models" / f"{run_dir.name}.model"
    if not model.is_file():
        raise RuntimeError(f"Saved child model is missing: {model}")
    return {**{key: value for key, value in spec.items() if key != "command"},
            "run_directory": str(run_dir), "training_log": str(logs[-1]),
            "model": str(model.resolve()), "metrics": endpoint}


def rank_key(result: dict) -> tuple[float, float]:
    metrics = result["metrics"]["valid"]
    return (metrics["force_rmse_mev_per_a"], metrics["energy_rmse_mev_per_atom"])


def promote_winner(incumbent: dict, winner: dict, loser: dict, round_number: int, hashes: dict[str, str]) -> None:
    lineage_path = campaign.OUT / "model_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    branch = campaign.active_branch(lineage)
    batch_number = 1 + sum(
        float(item.get("pressure_gpa", -1)) == PRESSURE_GPA
        for item in campaign.branch_history(lineage)
    )
    entry = {
        "pressure_gpa": PRESSURE_GPA, "parent": str(incumbent["model"]),
        "model": winner["model"], "utc": utc(), "batch_number": batch_number,
        "batch_epochs": EPOCHS, "learning_rate": winner["lr"],
        "finetune_id": winner["finetune_id"], "branch_id": branch,
        "autonomous_round": round_number, "selection_metric": "saved_model_validation_force_rmse_mev_per_a",
        "validation_force_rmse_mev_per_a": winner["metrics"]["valid"]["force_rmse_mev_per_a"],
        "losing_child": loser["finetune_id"], "train_sha256": hashes["cumulative_train.extxyz"],
        "validation_sha256": hashes["random_validation.extxyz"], "split_sha256": hashes["random_split.json"],
    }
    lineage["history"].append(entry)
    lineage["current_model"] = winner["model"]
    campaign.write_json(lineage_path, lineage)
    campaign.write_json(TRAINING / "batch_complete.json", entry)


def initial_incumbent() -> dict:
    """Select the starting parent from the active lineage branch.

    Reads model_lineage.json and picks the most recent completed batch on the
    active branch. Falls back to the configured foundation model at lr=1e-4
    when no completed batch exists (i.e., first run after a foundation restart).
    """
    DEFAULT_LR = 1e-4
    lineage_path = campaign.OUT / "model_lineage.json"
    if lineage_path.exists():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        active = str(campaign.active_branch(lineage))
        history = [item for item in lineage.get("history", [])
                   if str(item.get("branch_id", "legacy")) == active]
        if history:
            latest = history[-1]
            model = Path(latest["model"])
            if model.is_file():
                lr = float(latest.get("learning_rate", DEFAULT_LR))
                if not math.isfinite(lr) or lr <= 0:
                    raise RuntimeError(f"Latest lineage batch has invalid learning rate: {lr}")
                return {
                    "model": model, "learning_rate": lr,
                    "run_name": latest.get("finetune_id", str(model.name)),
                    "metrics": {"valid": {
                        "force_rmse_mev_per_a": latest.get("validation_force_rmse_mev_per_a", float("inf")),
                    }},
                }
            print(f"[autonomous-finetune] Latest lineage model missing at {model}; falling back to configured foundation", flush=True)
    foundation = campaign.config()["finetune"]["initial_parent"]
    print(f"[autonomous-finetune] No completed model on active branch; starting from {foundation} at lr={DEFAULT_LR:g}", flush=True)
    return {"model": Path(foundation), "learning_rate": DEFAULT_LR, "run_name": foundation, "metrics": {}}


def main() -> None:
    args = parse_args()
    incumbent = initial_incumbent()
    preview = child_specs(1, Path(incumbent["model"]), float(incumbent["learning_rate"]))
    print(f"Incumbent: {incumbent['run_name']}\nmodel: {incumbent['model']}\nrecorded lr: {incumbent['learning_rate']:g}")
    for child in preview:
        print(f"GPU {child['gpu']}: lr={child['lr']:g}, epochs={EPOCHS}\n  {shlex.join(child['command'])}")
    if args.dry_run:
        print("DRY RUN: no dataset, status, dashboard, log, or process was changed.")
        return

    campaign.OUT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another autonomous fine-tuning controller holds {LOCK}") from exc

    round_number = 0
    controller_state = {"schema_version": 1, "state": "preparing", "started_utc": utc(),
                        "rounds": [], "incumbent": {
                            "model": str(incumbent["model"]), "learning_rate": incumbent["learning_rate"],
                            "run_name": incumbent["run_name"]}}
    campaign.write_json(STATE, controller_state)
    try:
        hashes = prepare_data(Path(incumbent["model"]), float(incumbent["learning_rate"]))
        controller_state.update({"state": "running", "data_sha256": hashes})
        campaign.write_json(STATE, controller_state)
        while args.max_rounds == 0 or round_number < args.max_rounds:
            round_number += 1
            verify_frozen_data(hashes)
            specs = child_specs(round_number, Path(incumbent["model"]), float(incumbent["learning_rate"]))
            append_experiment_note([
                f"ANALYSIS {datetime.now().astimezone().isoformat()}",
                f"- evidence: Autonomous round {round_number} uses parent {incumbent['model']}, recorded incumbent lr={incumbent['learning_rate']:g}, epochs={EPOCHS}, train_sha256={hashes['cumulative_train.extxyz']}, validation_sha256={hashes['random_validation.extxyz']}, split_sha256={hashes['random_split.json']}.",
                "- diagnosis / uncertainty: The prior validation-force plateau leaves learning-rate scale as the isolated optimization uncertainty.",
                f"- next one-factor test: Launch exactly two sibling children concurrently: GPU 0 at lr={specs[0]['lr']:g} (2x) and GPU 1 at lr={specs[1]['lr']:g} (0.5x), both for 20 epochs from the same parent and split.",
                "- why this test is informative: Bracketing the incumbent learning rate tests opposite step-size directions while holding model, data, split, optimizer, loss weights, precision, scheduler/SWA policy, and epoch count fixed.",
                f"- changed factor relative to the incumbent: learning_rate only ({incumbent['learning_rate']:g} to {specs[0]['lr']:g} or {specs[1]['lr']:g}).",
            ])
            write_status("running", incumbent, round_number, specs, started_utc=utc())
            campaign.event("autonomous_finetune_round_started", PRESSURE_GPA,
                           [TRAINING / "cumulative_train.extxyz", TRAINING / "random_validation.extxyz"], [],
                           parent_model=str(incumbent["model"]), incumbent_learning_rate=incumbent["learning_rate"],
                           children=[{"gpu": child["gpu"], "lr": child["lr"], "finetune_id": child["finetune_id"]} for child in specs])
            processes = [subprocess.Popen(child["command"], cwd=D2, start_new_session=True) for child in specs]
            while any(process.poll() is None for process in processes):
                time.sleep(args.interval)
                write_status("running", incumbent, round_number, specs, started_utc=controller_state["started_utc"])
            return_codes = [process.returncode for process in processes]
            if any(code != 0 for code in return_codes):
                write_status("failed", incumbent, round_number, specs, return_codes=return_codes)
                append_experiment_note([
                    f"ANALYSIS {datetime.now().astimezone().isoformat()}",
                    f"- evidence: Autonomous round {round_number} child return codes were {return_codes}; no child was promoted.",
                    "- diagnosis / uncertainty: At least one sibling trial failed, so the paired learning-rate comparison is incomplete.",
                    "- next one-factor test: Diagnose and rerun this unchanged paired round after correcting only the execution failure.",
                    "- why this test is informative: A complete sibling pair is required for an unbiased tournament decision.",
                    "- changed factor relative to this run: none proposed until the execution failure is identified.",
                ])
                raise RuntimeError(f"Autonomous round {round_number} failed with return codes {return_codes}")
            results = [saved_model_result(spec) for spec in specs]
            winner, loser = sorted(results, key=rank_key)
            # Saved-model tables do not include the pre-training parent. Read it
            # from either child's registry-style epoch series in its log.
            from finetune_megadataset import metrics as observed_metrics
            parent_force = observed_metrics(Path(winner["run_directory"]))["initial"]["force_rmse_mev_per_a"]
            winner_force = winner["metrics"]["valid"]["force_rmse_mev_per_a"]
            improved = winner_force < parent_force
            round_record = {"round": round_number, "parent_model": str(incumbent["model"]),
                            "parent_learning_rate": incumbent["learning_rate"], "parent_force_rmse": parent_force,
                            "children": results, "winner": winner["finetune_id"] if improved else None,
                            "improved": improved, "completed_utc": utc()}
            controller_state["rounds"].append(round_record)
            campaign.event("autonomous_finetune_round_completed", PRESSURE_GPA, [],
                           [Path(result["model"]) for result in results], winner=round_record["winner"],
                           parent_force_rmse_mev_per_a=parent_force, winner_force_rmse_mev_per_a=winner_force)
            append_experiment_note([
                f"ANALYSIS {datetime.now().astimezone().isoformat()}",
                f"- evidence: Round {round_number} saved-model validation force RMSEs were GPU 0={results[0]['metrics']['valid']['force_rmse_mev_per_a']} meV/A at lr={results[0]['lr']:g} and GPU 1={results[1]['metrics']['valid']['force_rmse_mev_per_a']} meV/A at lr={results[1]['lr']:g}; the common parent evaluated at {parent_force} meV/A.",
                f"- diagnosis / uncertainty: {'The lower-RMSE child improves the parent and is promoted.' if improved else 'Neither completed child improves the parent, so further deterministic repeats from this incumbent are not justified.'}",
                f"- next one-factor test: {'Use the winning child as parent and bracket its lr by 2x and 0.5x in the next 20-epoch round.' if improved else 'Stop autonomous fine tuning for human review.'}",
                "- why this test is informative: Promotion uses the same frozen validation set and compares saved checkpoints only after the complete 20-epoch sibling group.",
                f"- changed factor relative to this run: {'learning_rate only in the next sibling pair.' if improved else 'none; controller stopped.'}",
            ])
            if not improved:
                controller_state.update({"state": "completed_no_improvement", "completed_utc": utc()})
                campaign.write_json(STATE, controller_state)
                write_status("completed", incumbent, round_number, results, stop_reason="neither_child_improved_parent")
                return
            promote_winner(incumbent, winner, loser, round_number, hashes)
            incumbent = {"model": Path(winner["model"]), "learning_rate": winner["lr"],
                         "run_name": winner["finetune_id"], "metrics": winner["metrics"]["valid"]}
            controller_state["incumbent"] = {"model": str(incumbent["model"]),
                                               "learning_rate": incumbent["learning_rate"],
                                               "run_name": incumbent["run_name"]}
            campaign.write_json(STATE, controller_state)
        controller_state.update({"state": "completed_max_rounds", "completed_utc": utc()})
        campaign.write_json(STATE, controller_state)
        write_status("completed", incumbent, round_number, [], stop_reason="max_rounds_reached")
    except KeyboardInterrupt:
        for process in locals().get("processes", []):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        controller_state.update({"state": "interrupted", "updated_utc": utc()})
        campaign.write_json(STATE, controller_state)
        write_status("interrupted", incumbent, round_number, locals().get("specs", []))
        append_experiment_note([
            f"ANALYSIS {datetime.now().astimezone().isoformat()}",
            f"- evidence: Autonomous fine tuning was interrupted during or after round {round_number}; owned child process groups received SIGINT.",
            "- diagnosis / uncertainty: The round is incomplete and no partial child is eligible for promotion.",
            "- next one-factor test: Inspect both immutable run outcomes before deciding whether to repeat the unchanged round.",
            "- why this test is informative: It prevents promotion from an unpaired or incomplete comparison.",
            "- changed factor relative to this run: none.",
        ])
        raise
    except Exception as exc:
        for process in locals().get("processes", []):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        controller_state.update({"state": "failed", "updated_utc": utc(),
                                 "error": f"{type(exc).__name__}: {exc}"})
        campaign.write_json(STATE, controller_state)
        if (TRAINING / "random_split.json").is_file():
            write_status("failed", incumbent, round_number, locals().get("specs", []),
                         error=f"{type(exc).__name__}: {exc}")
        append_experiment_note([
            f"ANALYSIS {datetime.now().astimezone().isoformat()}",
            f"- evidence: Autonomous fine tuning failed during or after round {round_number}: {type(exc).__name__}: {exc}",
            "- diagnosis / uncertainty: The paired comparison is incomplete or could not be audited, so no additional child is eligible for promotion.",
            "- next one-factor test: Inspect the recorded child outcomes and correct only the identified execution or audit failure before repeating the unchanged pair.",
            "- why this test is informative: It preserves the requirement that both 20-epoch siblings complete and remain comparable before selection.",
            "- changed factor relative to this run: none until the failure is diagnosed.",
        ])
        raise


if __name__ == "__main__":
    main()
