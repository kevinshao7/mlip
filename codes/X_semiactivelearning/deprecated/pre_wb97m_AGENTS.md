# X semi-active-learning campaign rules

This is a new, self-contained pressure campaign. New runs start from foundation
`polar-1-l`; `polar-1-s` is retired and must not be used for new simulations or
training. Only an approved campaign fine-tune may become the parent of a
later campaign MD run.

All executable source and generated launch code belongs under
`mlip/codes/X_semiactivelearning/`. Keep only generated scientific outputs,
manifests, trajectories, logs, checkpoints, and provenance under
`mlip/outputsfull/X_semiactivelearning/`.

Run no stage by directly calling a legacy B/C/D script.  Use `campaign.py` so
that every action is recorded in `outputsfull/X_semiactivelearning/`.

The pressure ladder is 10, 17.7828, 31.0, 56.2341, and 100 GPa at NH3/H2O
R1.  Temperatures and target densities are interpolated in log10(pressure)
from the existing Uranus profile. The default is one MD seed per condition.
The user-approved P31 deployment may instead use two explicitly seeded,
independent replicas on GPU 0 and GPU 1; keep every other MD setting identical
between those replicas and preserve their trajectories separately.

Species analysis uses nearest-heavy-atom H assignment, exactly as the existing
B production analysis.  The only base (allowed) species are H2O, HO, H3O, NH3,
NH2, and NH4. The inherited formatter spells the ammonia family as H3N, H2N,
and H4N; these are the same allowed species. Any other complete component is a
non-base (illegal) species and is a diagnosis-only marker — it is never included
in DFT/training data.

Each DFT/training cluster contains one allowed centre molecule plus its five
nearest allowed complete molecular neighbours under PBC (six molecules total).
Cluster selection is **unbiased uniform frame sampling** — not event-biased.
Non-base species are used only to determine eligibility (must be present but
below 5% complete-component fraction); they do not anchor or bias cluster
selection.  Generate diagnostic/acquisition data only when non-base species
exist and their complete-component fraction is below 5%; otherwise flag the
condition for human review.

Each MD frame contains ~100 allowed molecules (50 NH3 + 50 H2O at R1), so
multiple independent clusters can be extracted from the same frame — each with
a different centre molecule.  This is both valid and encouraged: a 352-frame
trajectory that would yield only 352 clusters under single-centre-per-frame
sampling can yield up to ~35,200 distinct clusters.  The deduplication key is
`(trajectory_id, frame, center_component)`, not the frame alone.

PBC correctness is critical when extracting multiple clusters from the same
frame.  Every cluster must use its own centre molecule's first atom as the PBC
anchor for computing neighbour distances — never reuse an anchor from another
cluster or another molecule in the same frame.  Using the wrong anchor shifts
the wrapped positions of all neighbours, producing a different (and incorrect)
neighbour list.  Concretely: compute `delta = positions - anchor` wrapped into
the unit cell, then sort by `|delta|`; the five closest allowed complete
molecules (excluding the centre itself) are the neighbours.

No illegal species (non-base components) may appear as the centre or as any
neighbour in a DFT/training cluster.  If any of the five nearest neighbours is
an illegal component, skip to the next allowed molecule in the sorted list until
six allowed molecules have been collected.

Before selecting DFT clusters, inspect the automatically generated
`analysis/pressure_volume_time_series.png`. It plots raw virial pressure and
cell volume against production time; a visible volume drift or sustained offset
from the target pressure means the frames are not an equilibrium training set.

Never automatically decide that a model needs fine tuning.  A diagnostic set
of at most 10 uniformly sampled clusters is DFT-labelled and reviewed through
its parity plot and species plot.  An energy error around 50 meV/atom is a
guide for human review, not an automatic threshold.  Only `campaign.py approve`
may authorize the full acquisition set.  Record the scientific reason in that
approval.

Always collect and evaluate parity from every successfully validated DFT frame,
including a set smaller than the 30-frame fine-tuning minimum. The minimum
only gates training; it never suppresses validation diagnostics.

Fine-tuning training data must come only from validated campaign DFT outputs
under `outputsfull/X_semiactivelearning/cycles/*/dft/acquisition*/`. Do not
merge a legacy D2 or other external DFT training set; Polar-1-L is the model
initialization, while fixed validation/test files are evaluation-only.

The user-approved learning-rate tournament in `autonomous_finetune.py` is the
explicit exception to the historical single-GPU policy: it launches exactly
two 20-epoch siblings from one parent and frozen split, with GPU 0 at 2x the
incumbent learning rate and GPU 1 at 0.5x. It waits for both before comparing
saved-model validation force RMSE and promotes only the winner. Outside that
controller, autonomous fine tuning remains on CUDA device 1. Preserve every
attempt in training history and every promoted winner in model lineage.

The fixed D2 validation/test sets are never changed.  Validated newly acquired
DFT frames are appended to a cumulative campaign training pool.  Fine tuning
must name precisely one changed factor and preserve parent model/data hashes.

`provenance.jsonl` is append-only; do not edit it.  `campaign_summary.csv` and
each cycle's `manifest.json` are derived, human-readable indexes.  Preserve
raw MD, cluster, ORCA input/output, Slurm, and training artifacts.

Never `git commit` or `git push` any changes — all version-control actions
must be performed by the human operator.  Agents may read, write, and edit
files, but may not interact with git history or remote repositories.

On `dungeon1`, use `campaign.py dft ... --run-local` after preparation: it
runs one ORCA job using 20 of 24 CPU ranks through the environment's `orca_qc`
launcher after loading `mpi/openmpi-x86_64`, and skips normally terminated
outputs on restart. Do not launch multiple local ORCA jobs concurrently.
`--submit` is for a host with Slurm's `sbatch` available. Prepared inputs may
only be replaced with `--replace-prepared`, which archives them (including a
failed `.partial` output) and refuses if any ORCA output terminated normally.
Never change SCF parameters or retry a failed DFT calculation automatically.
The local runner preserves its `.out.partial`, classifies it, records it in
`dft_failure_report.json`, and continues with the next cluster. Only normal
ORCA terminations can be collected into training data.

For no-agent autonomy, use `autonomous.py` and change only its versioned
`autonomous_policy.json`. Do not add an LLM or agent decision point. GPU 0 is
the next-pressure MD worker; GPU 1 is reserved for parity/fine tuning. Never
idle GPU 0 waiting for DFT: immediately before each MD launch, read the latest
completed model from model lineage and record it in the MD manifest.

## Log locations

All autonomous-run logs live under `outputsfull/X_semiactivelearning/autonomous_logs/`
with the naming pattern `{stage}_{pressure}_{timestamp}.{log,exitcode}`.

| File | Meaning |
|------|---------|
| `.log` | Combined stdout/stderr of the subprocess |
| `.exitcode` | Single integer written by the subprocess on clean exit |
| Missing `.exitcode` | Process was killed externally (OOM, SIGKILL, power loss) |
| Exit code `-998` in `autonomous_state.json` | Sentinel: `.exitcode` file absent = external kill |

Per-cycle outputs:
- `cycles/{pressure}/md/` — trajectory `.xyz`, `manifest.json`, `run_md.py/sh`
- `cycles/{pressure}/analysis/` — species CSVs, `pressure_volume_time_series.png`
- `cycles/{pressure}/clusters/{kind}/` — `clusters.extxyz`, `selection.csv`, validation JSON
- `cycles/{pressure}/dft/{kind}/` — `manifest.csv`, `inputs/`, `outputs/`, `dft_failure_report.json`
- `provenance.jsonl` — append-only event log covering every campaign action

## Debugging checklist

Work through these steps in order before drawing any conclusions or taking corrective action.

**1. Identify what failed**
```
# Quick exit-code overview for all jobs in a session
python3 -c "
import json
s = json.load(open('outputsfull/X_semiactivelearning/autonomous_state.json'))
for t in s.get('completed_tasks', []):
    print(t['exit_code'], t['key'])
"
```
Exit 0 = success. Exit 1 = Python exception. Exit -998 = external kill (no `.exitcode` file).

**2. For a missing `.exitcode` (external kill)**
The process received SIGKILL from outside — most likely the Linux OOM killer or a
machine restart.  The log will end abruptly without a Traceback.
```
# Check dmesg for OOM events around the job's finish time
dmesg -T | grep -i "oom\|killed process" | tail -20
# Find last step reached in a tqdm log (progress written as \r segments)
python3 -c "
parts = open('autonomous_logs/md_P31p6228GPa_TIMESTAMP.log','rb').read().split(b'\r')
print(parts[-1].decode(errors='replace')[-300:])
"
```

**3. For a Python exception (exit 1)**
```
grep -n "Traceback\|Error\|Exception" autonomous_logs/{stage}_{pressure}_{timestamp}.log | tail -30
```

**4. Verify MD trajectory completeness before proceeding**
Always check frame count before running `analyze` or `clusters`.  A short trajectory
produces a non-representative training set.
```
python3 -c "
import ase.io, sys
frames = list(ase.io.iread('cycles/P31p6228GPa/md/*.xyz'))
print(f'{len(frames)} frames')
"
# Quick line-count check (n_atoms+2 lines per frame; 350-atom system = 352 lines/frame):
python3 -c "lines=sum(1 for _ in open('cycles/P31p6228GPa/md/production_*.xyz')); print(lines//352, 'frames')"
```
Target: 2000 frames (100,000 production steps ÷ save_interval 50).
Each frame holds ~100 allowed molecules, so 2000 frames = up to ~200,000
distinct (frame, centre) pairs available for cluster extraction.

**5. Diagnose DFT failures**
```
python3 -c "
import json
r = json.load(open('cycles/P31p6228GPa/dft/acquisition/dft_failure_report.json'))
from collections import Counter
print(Counter(v.get('classification','unknown') for v in r.values()))
"
```
High SCF-non-convergence fractions (>20%) suggest strained or unphysical cluster
geometries.  Do not retry or change SCF settings automatically.  Report to the
human operator with the failure report path.

**6. Check for duplicate clusters before DFT**
Deduplication key is `(trajectory_id, frame, center_component)`.
```
python3 -c "
import pandas as pd
df = pd.read_csv('cycles/P31p6228GPa/clusters/acquisition/selection.csv')
dup = df[df.duplicated(['trajectory_id','frame','center_component'], keep='first')]
print(len(dup), 'duplicates'); print(dup[['cluster_index','trajectory_id','frame','center_component']])
"
```
Drop later-index duplicates; archive originals before modifying any file.

**7. Inspect pressure/volume stability**
Open `cycles/{pressure}/analysis/pressure_volume_time_series.png` before approving
acquisition.  Visible volume drift or sustained pressure offset → do not select
clusters from those frames; report to the human operator.
