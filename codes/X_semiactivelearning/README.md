# Semi-active-learning pressure campaign

This directory will hold a self-contained, cycle-addressed workflow for
increasing-pressure MACE-POLAR MD, trajectory analysis, cluster selection,
ORCA DFT, and target-only fine tuning.  It intentionally does not write into
the legacy B/C/D workflow directories.

All generated data will be placed under:

```text
../../outputsfull/X_semiactivelearning/
  cycles/<cycle-id>/{md,analysis,clusters,dft,training}/
  provenance.jsonl
  campaign_summary.csv
  campaign_timeline.csv
```

Each command will append an immutable, timestamped provenance event containing
the command, resolved settings, input/output paths, file SHA-256 hashes,
Git revision, and parent cycle/model identifiers.  ORCA inputs and processed
outputs will retain formally calculated charge and explicitly verified spin.

Stage reruns never delete prior results: existing `md`, `analysis`,
`clusters/<kind>`, and `training` directories are moved to
`cycles/<cycle-id>/history/<stage>/<UTC timestamp>/` before a replacement is
prepared. `campaign_timeline.csv` is the compact, chronological index; the
append-only `provenance.jsonl` contains full file paths and SHA-256 hashes.
Together with `model_lineage.json`, these provide the end-to-end sequence of
MD conditions, DFT data, parent models, and fine-tuned child models.

Every terminal semi-active fine-tuning batch also appends a Git-tracked row to
`../finetuning_megadataset.jsonl`: campaign data hashes/sizes, randomized split
provenance, hyperparameters, initial/final MACE loss/RMSE, timing, method, and
parent model are centralized in one registry shared by all fine-tuning codes.

Active-learning data are generated only when non-base complete molecular
components exist but remain below 5% of all complete components. Each DFT and
training cluster excludes the illegal event molecule and contains one central
allowed molecule plus its five nearest allowed complete neighbours (six total).
MD analysis writes pressure/volume versus
production-time equilibration diagnostics before cluster selection.
Cluster selection also produces separate 180° GIFs for the H2 diagnosis view
and for the actual six-molecule, allowed-only DFT input; inspect the latter
before generating ORCA jobs.

## Existing workflow adapters selected

- MD: `B_conditionsproduction/NPTMACEproduction_base.py`
- MD species analysis: `B_conditionsproduction/plot_species_time_series.py`
  and `plot_final_species_distribution.py`
- cluster and ORCA conventions: `C3_DFTproductionstopH2`
- target-only PolarMACE fine tuning: `D2_naive`

The campaign configuration has deliberately not been finalized until the
scientific and execution settings below are confirmed.  Those choices change
what constitutes a model failure and how an iteration is allowed to progress.

## Commands

## Operator boundary: production MD is manual

**You run every production MD job yourself.**  The campaign and autonomous
controller must never start, restart, or terminate production MD.  The MD
command below is deliberately a foreground command: run it yourself in the
terminal when you choose to begin a production condition.  The controller may
only process already-completed user-owned MD outputs (analysis, allowed-only
cluster selection, CPU DFT, and GPU fine tuning).

## One-line operator commands

Run these from `X_semiactivelearning`; every command is intentionally one
physical line so it can be pasted directly into the terminal.  `campaign.py`
is the underlying stage engine; **`autonomous.py` is the normal command for
everything after you finish a production MD.**

```bash
# Select the best completed 17.78-GPa checkpoint before preparing P31.
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python campaign.py promote-best --pressure 17.7827941004 --metric validation_loss
# Prepare the user-only P31 production MD (does not start it).
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python campaign.py md --pressure 31
# User-only production MD: this is never launched by autonomous.py.
bash /home/kevinsh/mlip/outputsfull/X_semiactivelearning/cycle2/cycles/P31p0000GPa/md/run_md.sh
# After that production job is finished, start the autonomous DFT/fine-tune loop.
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python autonomous.py --watch
# Read-only autonomous status/heartbeat check.
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python autonomous.py --status
# Refresh the read-only campaign dashboard.
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python campaign_dashboard.py
```

The controller sees the completed user MD, performs analysis and its
deterministic species/eligibility gate, then runs allowed-only cluster
selection, one serial 20-CPU ORCA queue, collection/parity, and 20-epoch GPU-1
continuation batches.  It never launches an MD.  Direct `campaign.py analyze`, `clusters`, `dft`, and
`finetune` commands are recovery/debug tools, not the standard Cycle2 route.

```bash
python campaign.py init
python campaign.py md --pressure 10 --run
python campaign.py analyze --pressure 10
python campaign.py clusters --pressure 10 --kind diagnostic
python campaign.py dft --pressure 10 --kind diagnostic --run-local
python campaign.py dft --pressure 10 --kind diagnostic --collect
python campaign.py approve --pressure 10 --reason "DFT diagnostic warrants 50 new labels"
python campaign.py clusters --pressure 10 --kind acquisition
python campaign.py dft --pressure 10 --kind acquisition --run-local
python campaign.py dft --pressure 10 --kind acquisition --collect
python campaign.py finetune --pressure 10 --run --changed-parameter new_dft_clusters
```

DFT preparation and execution are separate. On `dungeon1`, which has no
Slurm `sbatch` command, use `--run-local` after input preparation. It runs one
ORCA calculation at a time using 20 of 24 CPU ranks specified in `%pal`, leaving
four host cores available for scheduler, heartbeat, and system work, through the
site-supported `orca_qc` launcher available in the active environment, loads
the required `mpi/openmpi-x86_64` module, prints a line at start/completion of
every cluster, writes outputs atomically, and can
be safely re-run: normally terminated outputs are skipped. It deliberately
does not run concurrent ORCA jobs, because that would oversubscribe the 24
CPU ranks. `--submit` remains available on a Slurm submission host.

To replace a prepared-but-never-run or failed set (for example 13 early
candidates with the requested 50 candidates), regenerate with
`--replace-prepared`. The prior inputs and any failed `.partial` output are
timestamp-archived; the command refuses if an ORCA output terminated normally.

The campaign never changes SCF parameters to rescue a failed calculation. A
failed local ORCA job is retained as `.out.partial`, classified, recorded in
`dft_failure_report.json`, and skipped so the next cluster can run. Only
normally terminated calculations are collected into the DFT/training dataset.

`--run` is explicit for MD and fine tuning; without it their resolved launch
scripts/commands are generated. Every stage requires an explicit pressure and
refuses to overwrite completed artifacts. `campaign.py status` reports the
parent/data/model chain.

## Deterministic autonomous controller (no AI agent)

`autonomous.py` is a policy-driven state machine, not an AI/LLM agent. Its
versioned policy is `autonomous_policy.json`: equilibration is logged but does
not block clusters; routine fine tuning uses the parity energy MAE above
30 meV/atom gate; failed DFT jobs are preserved and
skipped without changing SCF inputs; and a pressure is never replayed after a
fine tune.

Parity validation is never gated by a training minimum: collection
always writes a parity-status record and makes a parity plot whenever at least
one DFT structure validates. The 30-label count controls fine tuning only.

The first campaign fine-tuning batch is a deliberate bootstrap exception: it
may start with any nonzero validated campaign DFT set. Subsequent batches use
the normal parity-error gate, preventing indefinite retraining on an unchanged
small dataset.

Fine-tuning training inputs are exclusively validated `processed.extxyz`
datasets under this project's own `outputsfull/X_semiactivelearning/cycles/`
root. Polar-1-s supplies the initial weights; legacy DFT training sets are not
merged into campaign training. Each fine-tuning data snapshot is shuffled with
a recorded deterministic seed and split into 70% training and 30% validation.
When new campaign DFT labels arrive, the enlarged snapshot receives a new
randomized split; `random_split.json` records the exact members and hashes.
The old fixed D2 validation/test files are never used by this campaign.

When the gate is met, GPU 1 performs a 20-epoch continuation batch from the
latest campaign checkpoint. At every batch boundary it re-evaluates parity on
the latest parent model and all available collected DFT data before deciding
whether another batch is justified. Each batch has its own immutable run ID,
data hash, parent model, child model, and provenance event.

GPU 0 is exclusively the next-pressure MD worker. GPU 1 is exclusively the
MACE parity/fine-tune/validation worker. The CPU worker handles analysis,
clusters, ORCA, and DFT collection. Every MD records the current model at its
own launch, so the pressure queue advances while prior DFT/fine tuning is
still running without obscuring lineage. Immediately before preparing each new
MD, the controller reads `model_lineage.json`; it uses the latest completed
fine-tuned model available at that exact launch time, without idling GPU 0.

For the current operator-controlled campaign, `controller_may_launch_md` is
disabled: the GPU-0 MD-worker description above is historical scheduling
design only.  **Start production MD yourself with the one-line command in the
Operator boundary section; do not use `autonomous.py` to launch production.**

```bash
# Inspect the next deterministic actions; writes nothing.
python autonomous.py --once --dry-run

# Reconcile, launch eligible work, and exit. Suitable for a user timer/cron.
python autonomous.py --once

# Persistent controller; retains state in outputsfull/X_semiactivelearning.
# It prints a scheduler/MD/DFT progress heartbeat every 5 seconds by default.
python autonomous.py --watch

# Optional: override the heartbeat/poll interval for this invocation.
python autonomous.py --watch --interval 2

# Read-only live/status check.  A durable copy of each status line is also in
# outputsfull/X_semiactivelearning/autonomous_logs/controller_heartbeat.log.
python autonomous.py --status
tail -f /home/kevinsh/mlip/outputsfull/X_semiactivelearning/autonomous_logs/controller_heartbeat.log
```

`autonomous_state.json` stores PIDs, commands, logs, and exit files;
`autonomous_logs/` retains task stdout/stderr. On restart, normally completed
outputs are detected and an exact live MD/ORCA worker is not duplicated even
if its short-lived launch-shell PID has changed. The controller
creates `autonomous_decisions.json` within each pressure cycle for every
equilibration and fine-tune gate.

While it is running, the controller also refreshes the dashboard every 30
seconds (configured by `dashboard_update_seconds`); this is read-only and
does not alter the user-owned MD policy.

### Human priority control during a run

`operator_priorities.txt` is reread on every controller poll. It is a
deterministic `key = value` file (not AI interpretation), so a researcher can
change the next DFT/fine-tuning task without restarting jobs. The current file
holds work at 17.7827941004 GPa while further DFT data are acquired and tested.

```text
active_pressure_gpa = 17.7827941004  # or `all`
hold_pressures_above_active = true
active_acquisition_kind = auto  # makes acquisition_r2, acquisition_r3, ...
continuous_acquisition = true   # stop by setting this false
continue_finetune = true
finetune_on_new_dft_only = true
note = free human context; retained in the file
```

With `finetune_on_new_dft_only = true`, each 20-epoch batch is tied to the
SHA-256 of the validated `processed.extxyz` snapshot. The controller will not
repeat training on unchanged P17 labels; a new completed/collected DFT result
creates a new snapshot eligible for one continuation batch. Every completed
batch records the hashes of all campaign DFT inputs in `model_lineage.json`.
Set it to `false` to keep GPU 1 continuously running successive 20-epoch
batches while the parity gate remains unmet, including while the CPU obtains
the next DFT round.

`active_acquisition_kind = auto` with `continuous_acquisition = true` is the
continuous-data mode. Once `acquisition`, `acquisition_r2`, etc. finishes its
ORCA queue, the controller selects a disjoint allowed-only round from the same
production trajectory, creates the next immutable `acquisition_r<N>` folders,
and continues. Stop it cleanly at the next polling cycle by changing
`continuous_acquisition = false`; no MD is launched by this mode.

## One-figure campaign dashboard

The read-only dashboard combines the whole loop into one PNG: pressure-cycle
stages, DFT requested/validated/SCF-skipped counts, parity MAE against the
fine-tuning trigger, immutable model lineage, and saved-model endpoint RMSE.
The endpoint panel shows energy and force RMSE separately for the randomized
campaign training data (dashed) and its 30% validation split (solid), making overfitting or domain
shift visible. It never launches, stops, or changes an MD/DFT/training job.

```bash
# Refresh once
python campaign_dashboard.py

# Keep the same figure current while the user runs MD and the controller runs
# DFT/fine tuning; default refresh is every 30 seconds.
python campaign_dashboard.py --watch
```

For Cycle2, the canonical dashboard is
`/home/kevinsh/mlip/outputsfull/X_semiactivelearning/cycle2/campaign_dashboard.png`.
Every named-run refresh also atomically mirrors that image to the conventional
top-level `.../X_semiactivelearning/campaign_dashboard.png`, records the source
in `active_dashboard.json`, and archives the former unnamed-run image under
`dashboard_history/`. This prevents the top-level path from silently showing
an old lineage.

## Fine-tuning stability

Campaign fine-tuning batches pass `--no-swa`.  The standalone D2 training
default begins Stage Two/SWA at epoch 15, which sharply increases both the
force weight and learning rate; that is unsuitable for the campaign's
20-epoch incremental batches.  This does not change the standalone D2 recipe.

If validation regresses, begin a clean branch from the original PolarMACE
foundation without deleting models, DFT labels, or prior lineage:

```bash
python campaign.py restart-foundation --reason "campaign-validation regression"
```

The dashboard then reports only the active branch; the complete earlier branch
is retained in `model_lineage.json`, `lineage_branches/`, and the provenance log.

## Fresh named loop with inherited DFT labels

Use a named run to restart fine-tuning from `polar-1-s` without touching an
earlier loop. The command below creates the independent `cycle2` lineage and
snapshots the legacy run's validated acquisition labels into it. Snapshots are
copied with checksums, not linked, so subsequent ORCA collection cannot change
the new loop's initial training set.

```bash
export SAL_RUN_ID=cycle2
python campaign.py init
python campaign.py inherit-dft
python campaign.py status
```

Start Cycle 2 fine-tuning (one command, from this directory):

```bash
SAL_RUN_ID=cycle2 /home/kevinsh/env/bin/python campaign.py finetune --pressure 17.7827941004 --run --changed-parameter cycle2_foundation_restart --change-note "Fresh cycle2 fine tune from polar-1-s using immutable prior campaign DFT snapshots; no Stage Two/SWA."
```

Use the same environment variable for every cycle2 command, including
`autonomous.py` and `campaign_dashboard.py`. Cycle2 begins with `polar-1-s`;
prior models are never considered parents.

To reuse safe unfinished P17 ORCA inputs in parallel with Cycle2 fine-tuning,
while retaining SCF failures as skipped, prepare and run the continuation:

```bash
SAL_RUN_ID=cycle2 python campaign.py reuse-dft --pressure 17.7827941004
SAL_RUN_ID=cycle2 python campaign.py dft --pressure 17.7827941004 --kind acquisition_r3 --run-local
```

The reuse step also imports the immutable P17 event ledger. When this reused
pool is exhausted, `acquisition_r4` selects only previously unused events from
the same production trajectory.
