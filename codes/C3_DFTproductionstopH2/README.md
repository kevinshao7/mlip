# C3_DFTproductionstopH2

`extract_condition_clusters.py` builds one combined DFT cluster XYZ from all
condensed condition-production trajectories in:

`mlip/outputsfull/B1_conditionsproduction_stride100_xyz`

Only `P100GPa_*` condition trajectories are processed. Every condensed frame
in those trajectories is eligible. Pressure-ramp, temperature-ramp, and
`production_hold` labels are not used as masks. Conditions are processed from
high pressure/high `R` to low pressure/low `R`, and frames are scanned
latest-first within each condition.

The replacement extractor searches H-H, O-O, and N-N closest-approach pairs.
All three pair types use the smaller of the ASE O-H and N-H covalent lengths and
are selected when:

```text
0.8 * min(O-H, N-H) <= distance <= 1.0 * min(O-H, N-H)
```

Every atom within `1.5 * max(O-H, N-H)` of either seed is forcibly included.
Starting from that full set, graph-connected atoms are recursively added using
`1.1 * max(O-H, N-H)` as the edge cutoff. Charge filtering is disabled by
default; `--max-abs-charge` enables an explicit filter. H-H, O-O, and N-N have
independent per-frame, per-condition, and global quotas.
Progress is printed every 50 scanned frames by default; change this with
`--progress-every`.

Default output:

- `mlip/outputsfull/C3_DFTproductionstopH2_ON/condition_production_ON_closest_approach.xyz`
- `mlip/outputsfull/C3_DFTproductionstopH2_ON/condition_production_ON_closest_approach_summary.csv`
- `mlip/outputsfull/C3_DFTproductionstopH2_ON/condition_production_ON_stage_windows.csv`

Run cluster extraction from the repository root:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C3_DFTproductionstopH2\extract_condition_clusters.py --workers 8
```

The defaults cap each pair type independently at 1000 total and 200 per
condition. Override with `--max-total-per-pair` and
`--max-per-condition-per-pair` if needed.

Useful one-condition test:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C3_DFTproductionstopH2\extract_condition_clusters.py --condition P100GPa_R1 --workers 8 --output-xyz .\mlip\outputsfull\C3_DFTproductionstopH2\test_P100GPa_R1_clusters.xyz --summary-csv .\mlip\outputsfull\C3_DFTproductionstopH2\test_P100GPa_R1_summary.csv --stage-csv .\mlip\outputsfull\C3_DFTproductionstopH2\test_P100GPa_R1_stage_windows.csv
```

## DFT Job Generation

`expand_dft_jobs.py` is a compatibility launcher for BlueHive, where `python`
may still point to Python 2. It re-execs the Python 3 implementation in
`_expand_dft_jobs_py3.py` when needed.

All ORCA inputs are written with multiplicity 1, regardless of the `spin`
metadata in the cluster XYZ.

It reads:

`mlip/outputsfull/C3_DFTproductionstopH2/condition_production_stopH2_clusters.xyz`

and writes ORCA `.inp` plus grouped Slurm `.slurm` files under:

`mlip/codes/C3_DFTproductionstopH2/expand`

Generate all frames:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C3_DFTproductionstopH2\expand_dft_jobs.py --clean --group-size 10
```

Regenerate only grouped Slurm scripts from existing `.inp` files:

```bash
python expand_dft_jobs.py --slurm-only --clean --group-size 10
```

On BlueHive, after `git pull`, run from `codes/C3_DFTproductionstopH2`:

```bash
python expand_dft_jobs.py --clean --group-size 10
for f in expand/C3_DFTprod_stopH2_group_*.slurm; do sbatch "$f"; done
```
