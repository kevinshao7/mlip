# B2 molecular-species time series

`plot_species_time_series.py` applies the exact analysis from
`B_conditionsproduction/plot_species_time_series.py` to the fine-grid B2
trajectories. It defaults to:

- input: `mlip/outputsfull/B2a_conditionsfinereduced`;
- conditions: `mlip/codes/B2_conditionsfine/production_manifest.csv`;
- output: `mlip/codes/B2_conditionsfine/species_time_series`.

The molecular definition is unchanged. Heavy atoms are connected using the
1.20-scaled covalent-radius rule, and every H is assigned unconditionally to
its nearest H, O, or N atom under periodic minimum-image distances. Each
connected component is counted by molecular formula.

The B2 output is intentionally grouped into exactly seven columns and plotted
lines: `H2O`, `H3O`, `HO`, `NH3`, `NH4`, `NH2`, and `illegal`. `illegal` is the
sum of every molecular component whose elemental composition is not one of the
six explicitly named species. The CSV uses the same seven-series grouping as
the plot.

For speed, each pressure/composition trajectory is represented by 100 frames
selected approximately uniformly from its first through last reduced frame.
The original reduced-frame indices are retained, so times still span the full
trajectory at 0.25 ps per reduced-frame interval. Across the 15 B2 conditions,
the default run analyzes approximately 1,500 frames rather than every frame.

Run every fine-grid condition:

```powershell
python .\mlip\codes\B2_conditionsfine\plot_species_time_series.py
```

Run one condition as a smaller check:

```powershell
python .\mlip\codes\B2_conditionsfine\plot_species_time_series.py --run-id P17p7828GPa_R0p5 --workers 8
```

The reduced frames are 0.25 ps apart. Plot and CSV times are read from each
frame's `source_time_fs` metadata rather than estimated from its sampled row
number. Thus the first frame is at 0.0025 ps and the final plotted point uses
the actual final saved-frame time recorded by the reducer. All reference
options, including `--workers`, `--batch-size`, molecular cutoffs, and plotting
thresholds, remain available through `--help`. This B2 adapter uses eight
parallel CPU worker processes by default; pass `--workers N` to override it.
