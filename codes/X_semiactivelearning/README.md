# X Semi-Active Learning Campaign

NH3/H2O R1 pressure campaign using PolarMACE (polar-1-s foundation).
All commands run from `mlip/codes/X_semiactivelearning/`.
All outputs live under `mlip/outputsfull/X_semiactivelearning/`.

---

## Current state (Oct 2026)

| Pressure | MD frames | DFT set | Status |
|----------|-----------|---------|--------|
| P10 (10 GPa) | 2000 | acquisition (collected) | fine-tuning done |
| P17 (17.78 GPa) | 2000 | acquisition r1–r3 (collected) | fine-tuning done |
| P31 (31.62 GPa) | **352** (incomplete — job killed) | acquisition (49 clusters, ~32–68% SCF fail) | **acquisition_r2 DFT running** |

**P31 MD needs to be re-run** to get the full 2000 frames (100 000 production
steps at save_interval=50).  The existing 352-frame trajectory is usable but
spans only 8.8 ps.

**Active job**: P31 `acquisition_r2` — 500 clusters, unbiased uniform-frame
sampling, all 500 ORCA inputs prepared and running locally.

---

## File map

| File | Role |
|------|------|
| `campaign.py` | **Core stage engine** — md, analyze, approve, clusters, dft, finetune, status |
| `campaign_config.json` | Pressure ladder, MD settings, DFT counts, cluster config |
| `campaign_dashboard.py` | Read-only PNG dashboard (never launches anything) |
| `autonomous.py` | Deterministic policy controller (DFT → collect → finetune loop) |
| `autonomous_policy.json` | Versioned policy for `autonomous.py` |
| `autonomous_finetune.py` | 2-GPU learning-rate tournament controller (P31-specific) |
| `launch_single_gpu.py` | Manual single-GPU fine-tune override (accepts `-lr`) |
| `operator_priorities.txt` | Hot-reloaded operator directives for `autonomous.py` |
| `workflow_species.py` | Molecular component parser (library used by campaign.py) |
| `orca_output_validation.py` | ORCA output validator (library used by campaign.py) |
| `AGENTS.md` | AI agent rules, log locations, debugging checklist |
| `CAMPAIGN_QUESTIONS.md` | Approved scientific design decisions |
| `dft_launches/` | **Generated** SLURM scripts (not source — produced by `campaign.py dft`) |
| `md_launches/` | **Generated** MD launch scripts (produced by `campaign.py md`) |
| `deprecated/` | Superseded one-off scripts retained for reference |

---

## Environment

Always activate the environment before running any command:

```bash
source /home/kevinsh/env/bin/activate
cd /home/kevinsh/mlip/codes/X_semiactivelearning
```

**`SAL_RUN_ID`**: selects a named output subdirectory.
- Base P31 data (current active campaign): `SAL_RUN_ID=` (empty or unset)
- Cycle2 P17 data: `SAL_RUN_ID=cycle2`

The base output root is `outputsfull/X_semiactivelearning/cycles/<pressure>/`.
With `SAL_RUN_ID=cycle2` it becomes `outputsfull/X_semiactivelearning/cycle2/cycles/<pressure>/`.

---

## Stage commands

All commands are single lines (no backslash continuations to avoid whitespace errors).

### MD (operator-only — never run by autonomous.py)

```bash
# Prepare MD launch scripts without starting (shows resolved settings)
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py md --pressure 31.6228

# Re-run P31 production MD to completion (2000 frames)
bash /home/kevinsh/mlip/codes/X_semiactivelearning/md_launches/default/P31p6228GPa/run_md.sh
```

### Analysis

```bash
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py analyze --pressure 31.6228 --workers 20 2>&1 | tee /home/kevinsh/mlip/outputsfull/X_semiactivelearning/autonomous_logs/analyze_P31_r2.log
```

Inspect `cycles/P31p6228GPa/analysis/pressure_volume_time_series.png` before proceeding.

### Approve (required before acquisition clusters)

```bash
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py approve --pressure 31.6228 --reason "reason here"
```

### Clusters

```bash
# Diagnostic set (10 clusters, no approval needed)
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py clusters --pressure 31.6228 --kind diagnostic --workers 20

# Acquisition round (50 default, or --count N; requires approval)
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py clusters --pressure 31.6228 --kind acquisition --count 500 --workers 20 2>&1 | tee /home/kevinsh/mlip/outputsfull/X_semiactivelearning/autonomous_logs/clusters_P31_r2.log
```

Each frame holds ~100 allowed molecules, so a 352-frame trajectory yields up
to ~35,200 distinct `(frame, centre)` pairs.  Multiple centres per frame are
supported; deduplication key is `(trajectory_id, frame, center_component)`.

### DFT

```bash
# Step 1: prepare ORCA inputs (inspect before running)
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py dft --pressure 31.6228 --kind acquisition_r2 2>&1 | tee /home/kevinsh/mlip/outputsfull/X_semiactivelearning/autonomous_logs/dft_prep_P31_r2.log

# Step 2: run ORCA locally (sequential, 20 MPI ranks, skips completed outputs on restart)
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py dft --pressure 31.6228 --kind acquisition_r2 --run-local 2>&1 | tee /home/kevinsh/mlip/outputsfull/X_semiactivelearning/autonomous_logs/dft_run_P31_r2.log

# Step 3: collect validated outputs into training pool
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py dft --pressure 31.6228 --kind acquisition_r2 --collect 2>&1 | tee /home/kevinsh/mlip/outputsfull/X_semiactivelearning/autonomous_logs/dft_collect_P31_r2.log
```

SCF failures are logged to `dft/acquisition_r2/dft_failure_report.json` and
skipped automatically.  Never change SCF parameters or retry failed jobs.

**Pressure note**: `31.6228` is the log-spaced value that maps to
`cycles/P31p6228GPa/`.  Do not use `31.0` (that maps to the separate
`cycle2/cycles/P31p0000GPa/` path from an earlier attempt).

### Fine tuning

```bash
# Autonomous 2-GPU LR tournament (GPU0=2x LR, GPU1=0.5x; dry-run first)
/home/kevinsh/env/bin/python autonomous_finetune.py --dry-run
/home/kevinsh/env/bin/python autonomous_finetune.py

# Manual single-GPU override (human chooses learning rate)
/home/kevinsh/env/bin/python launch_single_gpu.py
/home/kevinsh/env/bin/python launch_single_gpu.py -lr 1e-4
```

Fine-tuning uses only validated campaign DFT outputs from
`outputsfull/X_semiactivelearning/cycles/*/dft/acquisition*/`.
Never merge legacy D2 or external DFT sets.

### Campaign status and dashboard

```bash
SAL_RUN_ID= /home/kevinsh/env/bin/python campaign.py status
/home/kevinsh/env/bin/python campaign_dashboard.py
/home/kevinsh/env/bin/python campaign_dashboard.py --watch
```

---

## Autonomous controller (DFT → collect → finetune loop)

`autonomous.py` is a deterministic policy-driven state machine (no LLM).
It handles analysis, cluster selection, ORCA, collection, and fine tuning
after you complete production MD.  It never launches MD.

```bash
# Dry-run: inspect next actions, write nothing
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python autonomous.py --once --dry-run

# Single reconciliation pass then exit
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python autonomous.py --once

# Persistent controller (heartbeat every 5 s)
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python autonomous.py --watch

# Status check (read-only)
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python autonomous.py --status
tail -f /home/kevinsh/mlip/outputsfull/X_semiactivelearning/autonomous_logs/controller_heartbeat.log
```

`autonomous_state.json` stores PIDs, commands, logs, and exit codes.
`autonomous_logs/` retains all task stdout/stderr.

### Operator priority control

`operator_priorities.txt` is re-read on every poll cycle (hot-reload, no restart needed):

```text
active_pressure_gpa = 31.0        # or `all` to release hold
hold_pressures_above_active = true
active_acquisition_kind = auto    # auto-increments: acquisition, acquisition_r2, ...
continuous_acquisition = false    # set true for continuous data mode
continue_finetune = true
finetune_on_new_dft_only = true   # one batch per new validated DFT snapshot
note = free text for provenance readers
```

---

## DFT rules (dungeon1)

- Use `--run-local` (not `--submit`; no Slurm on dungeon1)
- 20 of 24 CPU ranks per job; never run concurrent ORCA jobs
- Skips normally terminated outputs on restart (safe to re-run after crash)
- Replace un-run or failed inputs with `--replace-prepared` (archives old inputs; refuses if any output terminated normally)
- Never change SCF parameters or retry failed calculations

---

## Cluster rules

- Centre molecule: any allowed complete species (H2O, HO, H3O, NH3, NH2/H2N, NH4/H4N)
- Neighbours: 5 nearest allowed complete molecules under PBC (6 total)
- PBC anchor: **centre molecule's own first atom** — each cluster independent
- Illegal species (non-base components): diagnosis-only, never in DFT/training data
- Dedup key: `(trajectory_id, frame, center_component)`
- Multiple clusters per frame are valid and encouraged

---

## Recovery and debugging

See `AGENTS.md` for the full debugging checklist, log locations, and procedures for:
- Diagnosing external kills (missing `.exitcode` → OOM/SIGKILL)
- Verifying MD frame counts before proceeding
- Checking DFT failure rates from `dft_failure_report.json`
- Detecting duplicate clusters

```bash
# Quick status of all autonomous job exit codes
python3 -c "import json; s=json.load(open('../../outputsfull/X_semiactivelearning/autonomous_state.json')); [print(t['exit_code'], t['key']) for t in s.get('completed_tasks',[])]"
```

---

## Historical context (Cycle2)

Cycle2 (`SAL_RUN_ID=cycle2`) was an earlier P17/P31 fine-tuning loop that ran
into validation regression and was paused.  The base campaign (no SAL_RUN_ID)
is the current active work.  Cycle2 commands are preserved below for reference.

### Cycle2 setup (completed, do not re-run)

```bash
export SAL_RUN_ID=cycle2
/home/kevinsh/env/bin/python campaign.py init
/home/kevinsh/env/bin/python campaign.py inherit-dft
/home/kevinsh/env/bin/python campaign.py status
```

### Cycle2 fine tuning from P17 (completed)

```bash
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python campaign.py finetune --pressure 17.7827941004 --run --changed-parameter cycle2_foundation_restart --change-note "Fresh cycle2 fine tune from polar-1-s using immutable prior campaign DFT snapshots; no Stage Two/SWA."
```

### Restarting from foundation (if validation regresses)

```bash
/home/kevinsh/env/bin/python campaign.py restart-foundation --reason "reason here"
```

Retains all prior lineage in `model_lineage.json` and `lineage_branches/`.
