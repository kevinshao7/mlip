# X semi-active learning — active workflow

This is the current campaign. Its default namespace is
`outputsfull/X_semiactivelearning/wb97m_step5/`; earlier scientific outputs remain intact. The old controller, launcher, policy and guide are in
Git history. Other dated workflows are historical studies or
supporting dependencies, not alternative default launchers.

## Code layout

Data is indexed by pressure and computation stage under
`outputsfull/X_semiactivelearning/P10GPa/`: `gpu_md_results/`,
`intermediate_cut_clusters/`, and `cpu_dft_results/`. The active portable
2,000-cluster dataset is selected by `partial_clusters_active.json` and can be
tracked in Git. Runtime code remains shared across pressures; it is not copied
into each data folder.

| Location | Purpose |
| --- | --- |
| `production/` | Active MD, DFT, training, controller, validation and dashboard implementations |
| `tools/` | Optional source-environment build and historical comparison/visualization utilities |
| `tests/` | Isolated preparation, scientific validation and CPU restart checks; no production jobs |
| `batch_launches/` | Current prepared P10 launch bundle, selected by `p10_batch_active.json` |

The four top-level Python files (`p10_batch.py`, `campaign.py`, `autonomous.py`
and `launch_single_gpu.py`) are stable command adapters. Existing launch commands
still work, including `python launch_single_gpu.py`. Edit their implementations
in `production/`. Configuration and the single `hyperparameters.txt` notebook
remain here. Deprecated controllers and retired launch bundles have been removed;
committed versions remain available in Git history. Scientific outputs are preserved.

Run all checks from this directory with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## For agents: avoid repeating setup and broad debugging

Start with [AGENTS.md](AGENTS.md) for the current entry point and a task-to-test
map. Use [DEBUGGING.md](DEBUGGING.md) for evidence paths, failure-specific next
steps, environment/Git notes and the smallest sufficient validation. The active
P10 plan overrides generic campaign defaults; historical notebook entries do
not select today's model, duration or launch scripts.

## Prepared 10 GPa batch

For the 2026-11-05 **2,000-cluster extraction from unfinished replicas**, follow
[PARTIAL_CLUSTER_EXTRACTION.md](PARTIAL_CLUSTER_EXTRACTION.md). It preserves
the original launch plan and snapshots only checkpoint-committed data before
review, selection and geometry audit.

[P10_BATCH.md](P10_BATCH.md) describes ten independent one-GPU **foundation
polar-1-l** replicas, each with a **1 ns production through resumable 24 h allocations**
(plus unchanged 4 ps ramps), and exactly 2,000 Viper DFT clusters (1,950
acquisition +50 evaluation). Runtime sizing is provisional, not a measured H200
benchmark. `p10_batch_active.json` points to the frozen, checkpoint-pinned plan.
Each helper invocation submits exactly ten independent jobs without dependencies.
Further checkpoint restarts require a separate launch decision. Whole-molecule PBC and reviewed-window quality
checks are required before DFT; the target-site method pilot remains pending.

## Scientific loop

1. Run periodic NPT MD from `polar-1-l` at 10 GPa, NH3/H2O R1; subsequent MD
   uses the accepted campaign model. Temperatures follow the existing Uranus
   profile interpolated in log pressure. Retain 0.5 fs steps, 2 ps pressure and
   temperature ramps and 2000 saved frames. The generic campaign default is
   37 ps; the active prepared P10 batch explicitly overrides it to 1 ns through
   resumable 24 h jobs. Read its frozen plan rather than applying the generic default.
2. Inspect pressure/volume stability and species plots. Record an acquisition
   approval with its evidence before cutting clusters. Select one allowed
   complete molecule plus five nearest allowed complete molecular neighbours
   under PBC; every cluster uses its own centre anchor. Uniformly sample frames,
   excluding illegal components, and deduplicate frame/centre pairs.
3. Run **dispersion-free wB97M/def2-TZVPD EnGrad RIJCOSX** on isolated clusters
   (PBC=False) through the Viper or BlueHive package. Do not use V or D4rev total
   labels. Recompute H-3N-2O charge and verify singlet multiplicity separately.
   Reject failed, inconsistent, stale, or nonfinite ORCA results.
4. Fine tune the campaign model using validated acquisition data only, with
   stable temporal train/validation assignments. Keep lr=1e-4, 20 epochs,
   no SWA and the shared training recipe; each acquisition changes data only.
5. Characterize energy MAE and force MAE/RMSE on **held-out current-pressure
   DFT clusters**. Compare the parent and saved child on the same snapshot.
6. At force error >=20 meV/Angstrom, collect another unique acquisition round
   and continue fine tuning at the same pressure. Preserve every attempt.
7. Advance by **+5 GPa** only for force-component **RMSE <20 meV/Angstrom**,
   with at least 30 validated held-out clusters and matching model/data hashes.
   The configured range is 10, 15, 20, ..., 100 GPa; 100 retains the prior upper
   endpoint. Energy error alone never advances pressure.

RMSE is the stated default interpretation of the user's force threshold.
It averages squared errors over Cartesian force components, in meV/Angstrom.
This is an adaptive validation gate, not an untouched final test. Consecutive
MD frames and adjacent blocks can remain correlated.

## Direct wB97M method pilot

ORCA's documented named wB97M keywords include dispersion. The proposed route
uses its explicit LibXC component block, without a D3/D4/NL keyword:

```text
! def2-TZVPD EnGrad RIJCOSX
%method
  Method DFT
  Functional hyb_mgga_xc_wb97m_v
end
```

The component's name contains `_v`; this does not authorize VV10 labels.
Output validation rejects evaluated dispersion terms. **This direct route has
not yet been run on the target ORCA installation.** Before production DFT, the
controller prepares one geometry from the audited C6 release. Run that pilot
on Viper/BlueHive, transfer its output, then verify against the C6 wB97M-only
energy/forces (1e-3 eV cluster energy and 1e-4 eV/Angstrom maximum force-component
tolerances). A mismatch blocks production; do not change the functional or
loosen tolerances automatically.

Reference: [ORCA 6.1 LibXC interface](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html).
C6's preserved subtraction labels remain supporting audit evidence; no old
D4rev-inclusive dataset is imported into the new campaign automatically.

## Commands

From `codes/X_semiactivelearning/`, in the GPU-capable Python environment:

```bash
python autonomous.py --once --dry-run
python autonomous.py --once
python autonomous.py --watch
```

The controller executes one stage at a time, stops on errors, and waits for
external Slurm DFT results. It does not automate copying files between hosts.
Run the controller directly on dungeon; DAIS GPU stages use the Slurm wrappers.
After analysis, inspect `cycles/P*/analysis/pressure_volume_time_series.png`
and `species_time_series.png`, then record the scientific decision:

```bash
python campaign.py approve --pressure 10 --reason "State pressure/volume stability evidence and accepted production window"
```

For the generic pressure-campaign namespace, once its acquisition data are
collected, the training launcher needs only:

```bash
python launch_single_gpu.py
```

This launcher defaults to `wb97m_step5`; it does not automatically select the
active named P10 batch. Do not start training on P10 data without selecting and
documenting that namespace as the prepared experiment. The launcher selects
the first unpassed pressure, audits E0 references, and
uses its cumulative campaign pool. It never selects the old acquisition_p10
D4rev package. Missing prerequisites fail explicitly. Each real launch creates
an independent datetime-named run under the active namespace's `runs/`.
The single experiment notebook is **`codes/X_semiactivelearning/hyperparameters.txt`**;
it records launches/outcomes and exact run, log, checkpoint and compiled-model
paths before optimization starts.

For DFT, generated inputs and per-site submitters live in
`dft_launches/wb97m_step5/.../{viper,bluehive}/`. Transfer both the prepared
`outputsfull` stage (manifest/inputs) and launch package to the chosen site;
run its `bash submit_all.sh`. It creates Slurm log directories before sbatch.
Each job preserves `.out`, per-job exit codes and ORCA scratch/property files.
Transfer results back to the same relative stage path. The controller waits
for every job's terminal record, excludes failed labels, and never retries SCF.
Site paths/module/MPI layouts are centralized in `site_profiles.json`.

DAIS wrappers, after creating `outputsfull/slurm` on DAIS:

```bash
sbatch finetune_dais.slurm
sbatch md_dais.slurm 10
```

They reuse gcc/14, cuda/12.8 and the working mlipenv, and preserve the assigned
GPU visibility. DAIS MD defaults to the first pressure if no pressure is passed;
its explicit pressure must match the current campaign stage. The CPU/GPU host
split and file transfers remain operator-owned. No simulation was launched
while preparing this workflow revision.

## Data partitions, references and tools

`workflow_policy.py` assigns complete 50-frame blocks: residue 0 modulo 6 is
held-out evaluation, residue 1 is validation, residues 2–5 are training. This
leaves an approximate 80/20 train/validation ratio within acquisition frames.
Membership stays fixed as more centres/rounds are added. The default acquisition
has 500 clusters and evaluation has 50; failed labels can reduce these counts.
Uniform sampling applies within each partition, with all evaluation frames
excluded from every acquisition centre.

`atomic_references.py` verifies the preserved C5 neutral H/N/O/S references,
including charge, multiplicity, method, normal termination and exactly zero
printed atomic D4 energy, before reusing the underlying wB97M E0s. Cluster labels
are absolute electronic energies in eV and forces in eV/Angstrom; no historical
atomization subtraction is applied to `REF_energy`.

| Tool | Role |
| --- | --- |
| `autonomous.py`, `autonomous_policy.json` | Sequential loop and force/pressure decision |
| `campaign.py`, `workflow_species.py` | MD, analysis, PBC cluster selection, collection and parity |
| `method_pilot.py`, `orca_output_validation.py` | Direct-method calibration, strict DFT validation |
| `site_dft.py`, `site_profiles.json` | Viper/BlueHive Slurm packages |
| `launch_single_gpu.py`, `training_bridge.py` | Active launcher and shared D2 recipe adapter |
| `atomic_references.py`, `atomic_references/` | Audited matching E0s and evidence |
| `../C6_wb97m_nodisp/` | Historical D4 subtraction and calibration labels |
| `../B_conditionsproduction/NPTMACEproduction_base.py` | Supporting MD template |

The large-model P10 trajectories documented in the old campaign are absent
locally and were compression transients. Local legacy polar-1-s P10 trajectories
are not this campaign's MD baseline. See the root `DATA_PROTOCOL.md` for the
inventory; preserve old files without importing them into the new training pool.

## Checks

```bash
python -m unittest discover -s tests -p test_active_workflow.py
python -m py_compile campaign.py autonomous.py launch_single_gpu.py training_bridge.py
```

CPU smoke tests establish orchestration/data checks, not ORCA route equivalence
or successful GPU optimization. Those require the pilot and actual site runs.
