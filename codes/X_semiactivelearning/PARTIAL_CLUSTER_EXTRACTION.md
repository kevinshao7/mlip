# Extracting clusters from the ten unfinished P10 replicas

This is an extraction-only branch of the active P10 workflow. The user's
2026-11-05 request reduced the requested budget from 5,000 to **2,000 total**:
1,950 acquisition and 50 evaluation clusters. It does not authorize new MD,
DFT or training jobs. Never mark incomplete MD complete to enable extraction.

The last successfully packaged extraction is selected by
`partial_clusters_active.json` in this workflow directory. It is separate from
the MD plan pointer. Resolve its exact workspace/file and verify its recorded
hash; do not repeat extraction when the requested dataset already exists.

## Pressure and computation-stage layout

The Git-ready release is under:

```text
outputsfull/X_semiactivelearning/P10GPa/
  gpu_md_results/                 index + local links to original replica data
  intermediate_cut_clusters/
    polar1l_2000_20261105T153411805462Z/
      clusters_2000.xyz           portable unlabelled geometry dataset
      clusters_2000_manifest.json portable per-cluster roles and source indices
      checksums.json
      provenance/                original audit, selection and analysis records
  cpu_dft_results/                index; DFT has not run for this dataset
```

Only the curated cluster release and small stage indexes/readmes are allowed
by Git ignore rules. Raw GPU MD/restarts/weights remain outside this package.
The GPU stage's local symlink preserves live restart paths; do not commit it.
Historical absolute paths in `provenance/` are audit evidence, not prerequisites
for loading the portable XYZ after cloning. Future CPU DFT results belong in
their own pressure/stage/site/run directory and must never be confused with
these unlabelled geometries.

After packaging an audited workspace, create the Git-ready copy with:

```bash
python tools/package_clusters.py --workspace "$WORKSPACE" --git-package
```

This verifies source dataset/audit hashes, copies only the compact dataset and
provenance, and updates `partial_clusters_active.json` to the portable file.
It leaves the original extraction workspace and raw MD directories intact.
For another dated release, add a narrow ignore-rule exception for that exact
curated folder; never unignore the full GPU/DFT output tree. Commit/push only
when requested. Check `git add --dry-run` before publishing.

## Locate the source and preserve it

Start with `p10_batch_active.json`, resolve its `plan`, then use the plan's
`output_relative`. Do not select the newest-looking run or use `autonomous.py`.
The named P10 `p10_batch.py analyze` deliberately requires completed 1 ns runs;
do not weaken that requirement to process partial data.

Use the existing implementation through `tools/extract_partial_clusters.py`.
It reuses `production/p10_batch.py` for frozen-plan/model verification,
`production/campaign.py` for analysis and cluster construction, and
`production/cluster_quality.py` for the final independent geometry audit.
No production source or launch-plan hash needs to change for this tool.

From the workflow directory, with the verified Python environment activated:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python tools/extract_partial_clusters.py prepare
```

Set these variables **before Python starts**, for analysis, extraction and
packaging. `--workers 4` limits worker processes, not numerical-library threads.
The first 2026-11-05 extraction oversubscribed roughly 56 CPU threads because
NumPy initialized before the existing campaign function set thread variables.
It was interrupted, the completed 50-cluster evaluation artifact was preserved
and independently audited, then acquisition resumed using single-threaded
numerical libraries. Do not kill an MD process to resolve extraction slowness.

Record the **exact workspace printed**. A new workspace is created under the
source run's `cluster_extractions/partial_2000_<UTC>/`. It contains the source
plan/config, ten snapshot folders and `snapshot_manifest.json` with input and
code SHA256 hashes. A prepare operation does not extract clusters.

The tool copies each atomic `restart.npz`, then copies trajectory/thermo bytes
only through its committed offsets. It verifies frame counts against checkpoint
steps, original 0.5 ps saving cadence, composition H250N50O50, finite geometry,
positive cell volume, seeds, model hash, batch-config hash and physical metadata.
Uncommitted tails and incomplete final frames are excluded without editing raw
files. A short committed prefix fails; never silently salvage a damaged prefix.
Snapshot files are hash-checked before each later stage. Run from a fresh
workspace after code changes; never update recorded hashes to bypass a mismatch.

## Analyze and review before extraction

Replace `$WORKSPACE` below with the exact printed path; do not choose it by glob.

```bash
python tools/extract_partial_clusters.py analyze --workspace "$WORKSPACE" --workers 4
```

Review `cycles/P10p0000GPa/analysis/` inside that workspace:

- `thermo_pressure_volume.png` and `thermo_review.json`: actual MD pressure
  including the kinetic contribution, original production times, volume and
  four chronological block means per replica. Blocks are exploratory diagnostics,
  not independent samples or a proof of equilibration.
- `species_time_series.png`, `species_time_series.csv`, `events.csv` and
  `summary.json`: species eligibility and per-replica frame accounting. The
  existing threshold is strictly less than 5% illegal components; those
  components never become cluster centres or neighbours.
- The existing `pressure_volume_time_series.png` uses extxyz virial stress
  **without the kinetic pressure term**. Do not mistake it for the total NPT
  pressure recorded by the MD runner. Its plotted time resets at the selected
  start frame; the thermo review preserves the original production clock.

If an initial transient should be excluded, rerun analysis with an explicitly
reviewed zero-based frame index, for example `--start-frame N`. Source frame
indices remain unchanged. Frame 0 is production time 0.5 ps, so frame N is
time `(N+1)*0.5 ps`. Earlier analysis is archived by the existing pipeline.
Keep at least one frame in each replica and preserve eligible evaluation blocks.
The policy uses original frame blocks of 50, modulo six: evaluation residue 0,
validation residue 1, other blocks training. Excluding the first 50 frames of
these short runs may remove all evaluation frames; do not silently redefine
the split just to make a selection fit.

Record the user's authorization and the reviewed window/limitations before
extracting. Approval binds the exact `analysis/summary.json` hash. Never copy
an approval from another analysis or claim that partial trajectories are
equilibrated solely because their pressure average is near the target.

## Extract and audit

```bash
python tools/extract_partial_clusters.py extract --workspace "$WORKSPACE" --workers 4 \
  --reason "Record actual authorization, reviewed diagnostics, window and limitations"
```

The existing selector samples frames uniformly and allocates counts in proportion
to usable frames across the ten replicas. Multiple distinct centres can come
from one frame; 2,000 clusters do **not** mean 2,000 independent configurations.
All centres from evaluation frames remain excluded from acquisition. Validation
remains governed by the existing temporal-block policy, not a new random split.

Centre selection is sequential CPU work, followed by four construction workers.
The existing constructor collects all results before writing the cluster file,
so there can be several minutes without new output after `building ... clusters`.
Confirm that the exact extraction process/workers are using CPU rather than
relaunching a second extraction. A completed evaluation file does not mean the
acquisition phase or the final geometry audit has finished.

Each isolated, nonperiodic cluster has one allowed complete central molecule
and five nearest allowed complete molecular neighbours. Whole molecules are
unwrapped internally, then translated together through PBC. Intramolecular
distance errors must be at most 1e-8 Angstrom. Charge is explicitly H-3N-2O;
singlet multiplicity and even electron count are separately checked. These are
unlabelled geometries: no `REF_energy`/`REF_forces` or DFT results are invented.

Success requires all of:

- `cycles/P10p0000GPa/clusters/acquisition/clusters.extxyz`: 1,950 clusters;
- `cycles/P10p0000GPa/clusters/evaluation/clusters.extxyz`: 50 clusters;
- corresponding `selection.csv`, `cluster_validation.json` and preview galleries;
- workspace `cluster_quality.json`: `status=passed`, `total=2000`, unique
  `(replica, original frame, centre)` keys, correct frame partitions/window,
  complete molecule membership, finite geometry, charge/spin and preserved PBC
  geometry. Short contacts are reported; no new physical cutoff is imposed.

The tool refuses an existing extraction instead of overwriting it. Preserve
failed outputs and diagnose them. Raw MD/restarts, snapshots and the original
2,000-input P10 launch bundle remain intact. Do not point `prepare-dft` at the
original batch and assume it selects this separate extraction: DFT preparation
must explicitly consume and revalidate this workspace, with the existing
target-site method pilot still mandatory.

Package the audited split files into the single `.xyz` requested by the user:

```bash
python tools/package_clusters.py --workspace "$WORKSPACE"
```

This creates `clusters_2000.xyz` (extended XYZ syntax) and
`clusters_2000_manifest.json`. Acquisition records come first (zero-based indices
0–1949); evaluation records follow (1950–1999). Every record retains
`cluster_kind`, source replica/frame/centre, explicitly calculated charge and
multiplicity. Packaging verifies the audited file hashes and exact role counts,
rejects duplicate sources, and refuses overwrite. The `.xyz` suffix is compatible
with standard XYZ viewers; use ASE `format='extxyz'` when metadata matters.

Append exact artifact locations, counts, diagnostic evidence, uncertainty and
next action to the single workflow `hyperparameters.txt`. Future agents should
read that entry and the exact workspace manifest before rerunning any stage.

## Focused checks

```bash
python -m unittest discover -s tests -p 'test_partial_extraction.py'
python -m unittest discover -s tests -p 'test_cluster_quality.py'
```

These are snapshot safety and extraction-invariant checks, not production MD.
