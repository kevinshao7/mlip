# C3_DFTproductionstopH2

`extract_condition_clusters.py` builds one combined DFT cluster XYZ from all
condensed condition-production trajectories in:

`mlip/outputsfull/B1_conditionsproduction_stride100_xyz`

It uses copied Slurm `.err` files in:

`mlip/outputsfull/slurm`

to identify each condition's `production_hold` window. The first 10% of that
window is dropped. Conditions are processed from high pressure/high `R` to low
pressure/low `R`, and eligible condensed XYZ frames are scanned latest-first
within each condition.

If a condition has no copied `.err` log containing `production_hold`, the
extractor falls back to scanning that condition's full condensed XYZ instead of
skipping it. This keeps the high-pressure `P100GPa_*` trajectories active even
when their Slurm logs are absent.

The cluster logic is the current C3 stop-H2 logic: same-element H2/N2 candidate
pairs are selected when:

```text
formed_cutoff <= distance <= near_cutoff
```

Each accepted cluster is the union of the connected components grown separately
from the two seed atoms using:

```text
graph_cutoff = near_graph_cutoff_scale * seed_distance
```

Default output:

- `mlip/outputsfull/C3_DFTproductionstopH2/condition_production_stopH2_clusters.xyz`
- `mlip/outputsfull/C3_DFTproductionstopH2/condition_production_stopH2_clusters_summary.csv`
- `mlip/outputsfull/C3_DFTproductionstopH2/condition_production_stopH2_stage_windows.csv`

Run cluster extraction from the repository root:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C3_DFTproductionstopH2\extract_condition_clusters.py --workers 8
```

The default extraction cap is 2000 total clusters, with no more than 400
clusters contributed by any single condition. Override with
`--max-total-clusters` and `--max-clusters-per-condition` if needed.

Useful one-condition test:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C3_DFTproductionstopH2\extract_condition_clusters.py --condition P100GPa_R1 --workers 8 --output-xyz .\mlip\outputsfull\C3_DFTproductionstopH2\test_P100GPa_R1_clusters.xyz --summary-csv .\mlip\outputsfull\C3_DFTproductionstopH2\test_P100GPa_R1_summary.csv --stage-csv .\mlip\outputsfull\C3_DFTproductionstopH2\test_P100GPa_R1_stage_windows.csv
```

## DFT Job Generation

`expand_dft_jobs.py` is a compatibility launcher for BlueHive, where `python`
may still point to Python 2. It re-execs the Python 3 implementation in
`_expand_dft_jobs_py3.py` when needed.

It reads:

`mlip/outputsfull/C3_DFTproductionstopH2/condition_production_stopH2_clusters.xyz`

and writes ORCA `.inp` plus grouped Slurm `.slurm` files under:

`mlip/codes/C3_DFTproductionstopH2/expand`

Generate all frames:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\C3_DFTproductionstopH2\expand_dft_jobs.py --clean --group-size 10
```

On BlueHive, after `git pull`, run from `codes/C3_DFTproductionstopH2`:

```bash
python expand_dft_jobs.py --clean --group-size 10
for f in expand/C3_DFTprod_stopH2_group_*.slurm; do sbatch "$f"; done
```
