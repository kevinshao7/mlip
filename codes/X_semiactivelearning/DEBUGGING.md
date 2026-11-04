# X workflow debugging: find the stage before changing code

This guide is a lookup table. Start with AGENTS.md and the active plan, then
read only the section matching the symptom. Paths below are relative to the
repository unless stated otherwise. No command in this guide submits jobs.

## Two-minute orientation

```bash
git status --short --branch
cat codes/X_semiactivelearning/p10_batch_active.json
```

Resolve that JSON's `plan` path. In the plan, inspect `output_relative`,
`config_relative`, `source_sha256`, `foundation_checkpoint`, `replicas`, and
`md`. Do not choose a directory by name/date or assume root campaign defaults
apply to this named batch. The plan/config are the authority; the notebook is
append-only history. See P10_BATCH.md for the current 1 ns/independent-job setup.

On the verified DAIS checkout, set this once for the session:

```bash
MLIP_PY=/dais/fs/scratch/kshao/mlipenv/bin/python
```

The interpreter is Python 3.11. Bare python was absent and system python3 was
3.6. Do not reinstall dependencies or blame annotations on a workflow bug.
MACE v0.3.17 imports from the editable GitHub checkout `mace/mace/` through
mlipenv; aseMolec/ remains a placeholder. See [DAIS_ENVIRONMENT.md](DAIS_ENVIRONMENT.md)
for source commits, activation, build records and rollback.

## Where the evidence lives

Let `<out>` be the plan's `output_relative`; P10 is `cycles/P10p0000GPa`.

| Evidence | Location | Interpretation |
| --- | --- | --- |
| Frozen scientific configuration | active bundle `campaign_config.json`, copied to `<out>/campaign_config.json` | Hashes must agree; don't fall back to generic 37 ps defaults |
| Exact model identity | plan `foundation_checkpoint`, `<out>/model_binding.json`, `model/foundation_provenance.json` | Foundation polar-1-l; SHA-locked; weights intentionally ignored by Git |
| Job code | active bundle `md_replica_00_dais.slurm` through `09`, `submit_md_dais.sh` | Ten independent one-GPU requests; six separate restart jobs per replica |
| Job stdout/stderr | `<out>/slurm/md_replica_XX_<jobid>.{out,err}` | Start at the first traceback/error; keep failed logs |
| Physical progress | `<out>/cycles/P10p0000GPa/md/replica_XX/production_status.json` | Production time and complete flag, not just a Slurm exit code |
| Exact restart | same replica directory: `restart.npz`, `resume_code.json` | Full MTK state and generated-code hash; don't load arbitrary pickle |
| Replica completion | same replica directory: `complete.json` | Written only when the physical target is reached |
| Source trajectory/thermo | same replica directory: `production_*.xyz`, `*_thermo.txt` | May be partial; don't assume 2000 frames until checked |
| Sampling/ensemble review | cycle `analysis/summary.json`, pressure/volume and species PNG/CSV | Window, stride, replica IDs, eligibility; approval binds its SHA |
| Cluster sources and geometry | cycle `clusters/{acquisition,evaluation}/selection.csv`, `cluster_validation.json`, `clusters.extxyz` | Source keys and complete-molecule membership |
| Production geometry gate | `<out>/cluster_quality.json`, `dft_package.json` | All 2000 structures/hashes/window/partitions must pass |
| Method calibration | `<out>/method_pilot/`, `method_verified.json` | Actual direct-wB97M/ORCA equivalence; not a syntax test |
| Raw DFT and failures | cycle `dft/<role>/{inputs,outputs,artifacts}`, `collection.csv` | Per-input exit code alone doesn't prove normal termination/valid gradients |
| Training evidence | selected namespace `runs/`, cycle `training/`, `model_lineage.json` | Verify namespace and actual run/model existence before reporting a result |
| Continuous lab notebook | `codes/X_semiactivelearning/hyperparameters.txt` | Append concise evidence/diagnosis/next test/one factor; never replace |

A successful Slurm segment can mean **checkpointed pause**, not 1 ns completed.
The runner's exit 75 is handled as a clean pause by p10_batch.py. Read physical
progress/complete markers before calling a replica finished. The helper's six
segments are an estimate; no code should silently label an unfinished trajectory
complete or admit it to cluster extraction.

## Symptom → targeted action

| Symptom | Inspect next | Avoid |
| --- | --- | --- |
| `bwrap ... namespace ... ENOSPC` before a command runs | Tool sandbox namespace failure; use the permitted escalation path once and reuse approvals | Deleting project data, treating it as a Python/data error, repeating many sandbox failures |
| Python missing / future annotations syntax error | Use the known mlipenv interpreter; verify version once per host | Reinstalling an environment or rewriting supported Python syntax |
| Source/config/model hash mismatch | Exact mismatched path and active plan; whether a generator changed after preparation | Editing recorded hashes to bypass the freeze |
| Model binding absent after Git pull | `cache-foundation` prepares model bytes and provenance from the pinned URL/cache | Searching for a retired fine-tuned model; ten simultaneous downloads; committing weights |
| Slurm says complete but trajectory is short | `production_status.json`, `restart.npz`, segment logs | Rerunning thermalization/ramps or deleting the trajectory |
| Next segment won't run | That replica's dependency/job exit status and `resume_code.json` | Depending on all other replicas or submitting a combined ten-GPU job |
| Restart version/setting mismatch | ASE/Torch/model/config evidence in restart and frozen plan | Resetting chain state or weakening signature checks to get past the error |
| Broken molecules or changed neighbour ordering | `molecule_images`, original PBC pair distances, cluster component offsets | Atomwise MIC about a distant centre; reusing another cluster's anchor |
| Window changed / approval mismatch | Current analysis SHA and chosen start/stride; review plots, then record updated approval | Sampling from frame zero anyway or reusing an old window's approval |
| DFT method gate blocked | Actual pilot output and `verify-method-pilot` rejection reason | Fake `method_verified.json`, guessed !wB97M keyword, automatic tolerance changes |
| ORCA convergence failure or partial stdout | That output and collection issue/classification | Automatic SCF changes, retries or charge repair |
| Force gate will not advance | Current model/data hashes, held-out size, RMSE key/units | Energy MAE or in-training parity as advancement evidence |

The force gate uses `force_rmse_mev_per_angstrom < 20`, with >=30 valid held-out
clusters. `force_mae_ev_per_angstrom` is a different metric/unit. Exactly 20,
NaN, missing or stale metrics fail. Never mix V/D4rev total labels into this pool.

## Smallest sufficient verification

From `codes/X_semiactivelearning/`, using the known interpreter:

```bash
"$MLIP_PY" -m unittest test_p10_batch
"$MLIP_PY" -m unittest test_md_segments
"$MLIP_PY" -m unittest test_cluster_quality
"$MLIP_PY" -m unittest test_active_workflow.DFTValidationTests
```

Choose the module matching the edit, not all four by default. P10 submission
tests use a fake sbatch and prove independent predecessor chains. Restart tests
use tiny CPU fixtures and compare complete NPT state, not MACE/GPU production.
For generator edits also compile generated Python and `bash -n` the affected
shell files. For cross-cutting changes, run the full suite once after final edits:

```bash
"$MLIP_PY" -m unittest discover -s . -p 'test_*.py'
```

After passing focused checks, stop testing unless a new change/failure justifies
more. Documentation-only changes need local link/path checks and
`git diff --check`, not a new dated bundle, model download or simulation.

## Freeze/update procedure

Read the exact `source_sha256` list before editing executable code. A changed
hashed source requires a fresh dated plan via the generator, with old scientific
outputs preserved. Archive superseded code bundles and update the active pointer.
Reuse pinned model cache bytes. Do not modify a frozen plan to make an old
checkpoint run with changed physics; any restart migration needs an explicit,
reviewable diagnosis and versioned handling. Docs aren't hashed sources.

The available 266-frame C6 release is a pilot/reference artifact, not this
batch's future 2000 structures. Its recorded audit passed finite/charge/spin/PBC
checks, but parent MD is absent locally and historical whole-molecule preservation
is uncertified. The actual method pilot and new sampling eligibility remain
pending; do not repeat the basic audit unless the source hash or check changes.
Do not rescan the 16 GB legacy P10 trajectory just to rediscover its inventory.

## Git when explicitly requested

Check status/fetch/divergence first. Leave unrelated untracked checkpoints,
pt_output and conversion outputs untouched. Commit only the reviewed source and
notebook changes. On 2026-11-04 HTTPS fetch worked, its saved push token was
invalid, and existing SSH auth worked for this same repository:

```bash
git pull --ff-only origin main
git push git@github.com:kevinshao7/mlip.git main:main
git fetch origin
git rev-list --left-right --count HEAD...origin/main
```

Use those commands only when sync is authorized and history is understood;
resolve incoming overlap rather than forcing. Fetch after a URL-based push to
refresh origin/main. Don't inspect/print tokens or repeat an unnecessary login
flow when verified SSH already works. Explain any actual auth rejection.
