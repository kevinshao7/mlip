# C_DFTproduction Cluster Extraction

`extract_condition_clusters.py` builds DFT candidate clusters from the condensed
condition-production trajectories in:

`mlip/outputsfull/B1_conditionsproduction_stride100_xyz`

It uses the copied Slurm `.err` files in:

`mlip/outputsfull/slurm`

to identify the `production_hold` stage. The production-hold start/end are
converted to fractions of the logged total MD step count, then those fractions
are applied to the condensed XYZ frame range. This avoids relying on exact
one-to-one equality between `.err` step numbers and copied `.xyz` frame numbers.

The first 10% of the production-hold interval is dropped in this same fractional
coordinate.

Frame metadata is kept in the same fractional coordinate used for selection:

```text
source_frame_fraction = condensed_frame / (n_condensed_frames - 1)
```

The `.err` progress logs report MD steps, but stage filtering is done by
fractional progress. No exact condensed-frame-to-MD-step mapping is assumed.
The summary and extxyz metadata record `source_condensed_frame` and
`source_frame_fraction` for traceability.

Default output:

- `mlip/outputsfull/C_DFTproduction/condition_production_dft_clusters.xyz`
- `mlip/outputsfull/C_DFTproduction/condition_production_dft_clusters_summary.csv`
- `mlip/outputsfull/C_DFTproduction/condition_production_stage_windows.csv`

Run:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C_DFTproduction\extract_condition_clusters.py
```

The default is `--workers 8`. Each condition is scanned in a deterministic
streaming pass, then selected cluster builds are distributed across 8 worker
processes. This means a one-condition test still uses all 8 CPU cores during
cluster construction. Use `--workers 1` only for debugging.

Defaults target 20 conditions x 200 clusters = 4000 clusters:

- 100 isolated-H cut clusters per condition
- 100 random complete-fragment clusters per condition

One-condition test, keeping the full per-condition target of 100 isolated-H cut
clusters plus 100 unbiased random clusters:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C_DFTproduction\extract_condition_clusters.py --condition P0p11GPa_R0p1 --workers 8 --output-xyz .\mlip\outputsfull\C_DFTproduction\test_P0p11GPa_R0p1_clusters.xyz --summary-csv .\mlip\outputsfull\C_DFTproduction\test_P0p11GPa_R0p1_summary.csv --stage-csv .\mlip\outputsfull\C_DFTproduction\test_P0p11GPa_R0p1_stage_windows.csv
```

By default, each isolated-H sample is a cut cluster centered on an H atom whose
nearest O is farther than `--oxygen-exclusion-radius 1.7` from the seed H. The
initial environment radius is `--isolated-environment-radius 4.5`; recursive
completion then includes any atom within `--completion-radius 1.5` of already
selected atoms. The helper uses minimum-image distances with the periodic cell
before unwrapping the selected cluster into vacuum.

Random clusters are complete covalent fragments selected nearest to a random
seed atom until `--random-min-atoms 25` is reached, while never exceeding
`--random-max-atoms 60`.

All written clusters must have at least `--min-cluster-atoms 20` atoms. Smaller
candidate clusters are rejected, recorded in the summary CSV, and replaced from
the remaining candidate pool when possible. Runs print min/max accepted cluster
sizes per condition and overall.

For each condition, the written XYZ order is fixed: isolated-H cut clusters are
first, followed by unbiased random clusters.

## DFT job generation

`expand_dft_jobs.py` mirrors the BlueHive ORCA layout from
`codes/A_parityplot/8_5_bluehiveDFT`. It writes one ORCA `.inp` per cut
cluster and grouped Slurm scripts, each running 10 ORCA calculations
sequentially. It reads:

`mlip/outputsfull/C_DFTproduction/condition_production_dft_clusters.xyz`

and writes ORCA `.inp` plus grouped Slurm `.slurm` files under:

`mlip/codes/C_DFTproduction/expand`

Generate all frames:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C_DFTproduction\expand_dft_jobs.py --clean --group-size 10
```

On BlueHive, submit from `codes/C_DFTproduction` with:

```bash
for f in expand/C_DFTprod_cutcluster_group_*.slurm; do sbatch "$f"; done
```

The generated grouped Slurm scripts call `/software/orca/6.1.1/orca` directly
and do not call `module purge` or `module load`, avoiding the module-init
`unalias sudo` warning. Email notifications are enabled only for the first 10
and last 10 grouped Slurm files.

Each grouped Slurm script checks for completed `.out` files inside the per-STEM
loop. Do not add a pre-loop `OUTPUT_PATH` completion check: grouped jobs define
`OUTPUT_PATH` only after selecting the current STEM, and `set -u` will abort on
an unset `OUTPUT_PATH`.

The script prints status for:

- production-hold stage windows parsed from `.err` files
- conditions skipped because no `production_hold` was reached
- eligible-frame scans
- isolated-H candidate counts
- every 25 built isolated-H or random clusters per condition
