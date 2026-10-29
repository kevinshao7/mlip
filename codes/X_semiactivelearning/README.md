# X Semi-Active Learning Campaign

> [!IMPORTANT]
> **Foundation-model policy (2026-10-28):** Do not use `polar-1-s` for new
> simulations or training. Use the large MACE-POLAR checkpoint `polar-1-l`.
> Any `polar-1-s` references retained below describe immutable historical runs.

NH3/H2O R1 pressure campaign using the large PolarMACE `polar-1-l` foundation.
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

### Prepared P10 two-GPU, six-hour run

The named run `p10_2gpu_6h` preserves the completed default P10 campaign. It
uses two independent `polar-1-l` replicas at 10 GPa and 1920 K, one each on
GPU 0 and GPU 1. Each replica uses a 0.5 fs timestep, unchanged 2 ps pressure
and 2 ps temperature ramps, and a 45 ps production hold (98,000 total steps).
The 45-step save interval produces exactly 2,000 production frames per replica.

The six-hour target is an estimate scaled from the recorded 108,000-step run
that completed in 6 h 36 min; it is not a hard wall-clock cutoff. That earlier
throughput is not a `polar-1-l` benchmark, so the first large-model run should
be treated as the calibration for later six-hour jobs.

```bash
cd /home/kevinsh/mlip/codes/X_semiactivelearning/md_launches/p10_2gpu_6h/P10p0000GPa
/home/kevinsh/env/bin/python run_md_parallel.py
```

The official calculator spelling is `mace_polar(model="polar-1-l", ...)`; see
the [MACE-POLAR guide](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html)
and [MACE v0.3.16 release notes](https://github.com/ACEsuit/mace/releases/tag/v0.3.16).

The completed 2026-10-28 trajectories are not an equilibrium 10 GPa
acquisition pool. Across both 2,000-frame replicas, the mean virial pressure is
3.6717 GPa and the cell volume decreases from about 4148 to 2280 A^3. Even the
last 200 frames average only 5.85 and 5.88 GPa for replicas 0 and 1. After this
was reviewed, the user explicitly directed the 1,000-cluster extraction and
DFT preparation to proceed. Those clusters therefore retain a non-equilibrium,
sub-target-pressure caveat and must not be described as equilibrium 10 GPa
data. The evidence and override are recorded in the approval JSON,
`analysis/pressure_volume_time_series.png`, `analysis/summary.json`, and the
workflow-level `hyperparameters.txt`.

For this explicitly approved acquisition, ORCA 6 uses its documented
`wB97M-D4rev def2-TZVPD` simple-input keywords with `EnGrad RIJCOSX`. The named
run is configured for BlueHive ORCA 6.1.1 as 100 grouped Slurm jobs for a
1,000-cluster set (10 sequential clusters per job, 24 ranks each). Generated
Slurm logs remain under the run's `dft/<kind>/slurm_logs/` directory in
`outputsfull/`.

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

The active fine-tuning path accepts only frames tagged
`dft_functional_basis="wB97M-D4rev def2-TZVPD"` and refuses missing, mixed, or
historical wB97M-V provenance. Both campaign launchers explicitly select the
custom H/N/O/S E0 JSON produced by `C5_atomizationD4rev`; there is no active
fallback to foundation or C2 atomic energies. The four C5 atomic ORCA jobs are
complete, and the launcher verifies exact agreement between their CSV and JSON
before creating a run directory.

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


## Incoming P10 acquisition (2026-10-29)

From this workflow directory, run `python campaign.py prepare-acquisition` after
copying ORCA stdout into `outputsfull/acquisition/outputs`. The command checks
all 1,000 approved inputs from `dft_launches/p10_2gpu_6h/`, saves a dated audit
under `outputsfull/X_semiactivelearning/acquisition_p10/cycles/P10p0000GPa/dft/acquisition/snapshots/`,
and preserves every raw file. Missing, changing, incomplete, charge/spin,
geometry, method, or gradient-invalid outputs are excluded and listed in
`collection.csv`. `summary.json` records source hashes and counts. A rerun
creates a fresh snapshot; it never appends duplicate labels to the pool.

By default an incomplete transfer produces an audit only. After transfer is
complete, rerun the same command. `--allow-partial` explicitly permits preparation
from the validated subset (at least 30 labels). Presence of all files does not
imply all DFT calculations succeeded: inspect the rejection report.

Prepared data are this acquisition only, with a deterministic random 80/20
train/validation split; exact membership, seed, and dataset hashes are in
`training/random_split.json`. This is a structure-level split, not an
independent-trajectory generalization estimate; nearby MD frames can be
correlated. The transient/sub-target-pressure caveat is retained in extxyz.
Labels use isolated-cluster PBC=False, total energies in eV and forces in eV/A;
the shared training recipe retains its existing atomic reference convention.
No labels from earlier cycles or fixed D2 evaluation datasets are added.

Start the prepared experiment with `python launch_single_gpu.py`. Defaults:
polar-1-l, GPU 1, learning rate 1e-4, 20 epochs, no SWA. The old cycle2 launcher
is preserved in `deprecated/launch_cycle2_best_parent.py` for reference.
The campaign uses the existing training recipe through `training_bridge.py`;
new timestamped runs live under `outputsfull/X_semiactivelearning/acquisition_p10/runs/`.
Launch and outcome entries, including exact artifact paths, are mirrored to
this workflow's single `hyperparameters.txt` before optimization begins.
No training is started by the preparation command.

### GitHub-to-HPC dataset location

The portable dataset is **`outputsfull/X_semiactivelearning/acquisition_p10/hpc_dataset/`**
(relative to the repository root). It contains `processed.extxyz`, `train.extxyz`,
`validation.extxyz`, and `dataset.json` with relative filenames, SHA-256 hashes,
exact frozen split indices, and per-cluster acceptance/rejection results.
These four files have an explicit `.gitignore` exception. Raw ORCA stdout,
local audit snapshots, model caches, and training runs remain ignored.

During transfer this directory holds a **preview**, with
`ready_for_training: false`; the launcher refuses to train from it. Re-running
`python campaign.py prepare-acquisition` after copying updates the package and
sets readiness when all expected output files are present (failed DFT labels
remain excluded). An explicit `--allow-partial` can approve a partial snapshot.
Check rejection counts even when the package is ready.

After final preparation, add the four package files and changed workflow code
to Git, commit and push them, then pull on HPC. In the HPC environment, run:

```bash
cd codes/X_semiactivelearning
python launch_single_gpu.py
```

The launcher prints the resolved dataset path and split counts, verifies all
package hashes, restores the campaign acquisition using repository-relative
paths, and preserves the exported train/validation membership exactly across
Windows and Linux. Neither raw ORCA outputs nor local absolute paths are
required on HPC. The shared D2_naive training recipe and local MACE checkout
must be available in the repository/environment as before.

Finalized after the operator confirmed transfer completion on 2026-10-29:
363 outputs were received, 266 passed validation, 97 were rejected for missing
normal termination/energy/gradient, and 637 of the planned 1000 outputs were
absent. The available validated subset was finalized with --allow-partial:
213 training / 53 validation structures. dataset.json explicitly records
these counts and now has ready_for_training: true. The default HPC launcher
can use this package. This is a 266-label subset of the planned acquisition.
