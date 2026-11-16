# Separate foundation/SWA baseline

Launcher options are listed vertically, one option per line, in
`launch_single_gpu.py`. The2026-11-16 formatting update preserves every argument
and value; the original launcher/plan are under
`history/20261116T190658783762Z/`. A fresh prepared bundle refreshes source hashes
and output paths only; no optimization or job submission accompanied the update.

Prepared 2026-11-13 at the user's request. Starts from the original cached
polar-1-s, polar-1-m and polar-1-l foundations, each in a separate job. Existing training jobs,
low-LR pointers and launchers are unchanged. No submission occurred.

From this directory, polar-1-l is the default (use --model s or --model m
for the other sizes):

```bash
python launch_single_gpu.py
```

Preview without training:

```bash
python launch_single_gpu.py --dry-run
```

From the parent X workflow directory:

```bash
bash submit_swa_foundation.sh --dry-run
# Only after explicit launch authorization:
bash submit_swa_foundation.sh
```

The helper submits three independent jobs, one per size.
Each Slurm job requests one H200, 12 CPUs, 250000 MB and 24 hours on
gpu1, preserving assigned GPU visibility. A 24-hour allocation does not guarantee
completion; there is no automatic resubmission or modification of current jobs.

Explicit user recipe: energy/force weights10/1000, batch2/validation4, maximum
300epochs, SWA starting at150, scheduler patience15, early-stopping patience30,
evaluation interval4, EMA and SWA enabled, PerAtomMAE, float64, CUDA, seed123,
restart_latest and save_cpu. Label keys remain REF_energy/REF_forces with charge
and spin metadata. Architecture and cutoff come from the foundation.

Unspecified values come from installed MACE, not D2/workflow overrides:

| Option | Installed default |
| --- | --- |
| LR | 0.01 |
| SWA LR | 0.001 |
| EMA decay | 0.99 |
| SWA energy/force weights | 1000/100 |
| Weight decay | 5e-7 |
| Gradient clip | 10 |
| Optimizer / scheduler | adam / ReduceLROnPlateau |
| LR factor | 0.8 |
| Data-loader workers | 0 |

The actual command omits these options. MACE's normal loading path disables
multihead replay for this local PolarMACE file with no replay dataset, so its
multihead-only automatic LR1e-4/EMA0.99999 adjustment does not apply. No
force_mh_ft_lr override is supplied.

E0s are a hash-verified copy of atomic_references/e0s_wb97m.json: neutral
isolated-atom dispersion-free wB97M/def2-TZVPD energies in eV, never average or
foundation E0s. Labels remain absolute dispersion-free wB97M energies (eV) and
forces (eV/Angstrom). Fixed P10 split950train/1000validation is retained; all50
held-out evaluation structures remain excluded from optimization. The example's
5% split and TotEnergy/force keys do not override these audited data bindings.

active.json freezes the foundation hash, copied dataset/E0 hashes, upstream
defaults and relevant source hashes. Each actual invocation creates a new dated
run under its dedicated outputsfull directory and appends to the single X
hyperparameters.txt before optimization. No prior run/checkpoint is resumed;
restart_latest only searches the fresh child directory. This is a new recipe
baseline changing several factors, not an isolated LR/SWA experiment.

MACE patience can terminate before300epochs or move to stage two after a
stage-one plateau. Epoch150 is the configured transition, not a promise of150
completed stage-one epochs in the presence of early stopping. Both training
stages use their own loss weights; compare force/energy errors across the
transition rather than raw loss alone.

Training logs use <run_name>_run-123.log. Best checkpoints use
<run_name>_run-123_epoch-*.pt, with stage-two variants managed by MACE. Exported
models, if completed, are models/<run_name>.model and
models/<run_name>_stagetwo.model; compiled outputs, if supported, are
models/<run_name>_compiled.model and models/<run_name>_stagetwo_compiled.model.
Raw checkpoints and all logs remain preserved. No export/completion is claimed
by preparation or CPU dry-run.
