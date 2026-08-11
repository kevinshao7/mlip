# 8_7 MLIP-vs-DFT Histograms

Generate four histograms for one MLIP model against the BlueHive ORCA outputs:

```bash
python codes/A_parityplot/8_7_histogram/plot_mlip_dft_errors.py --model polar1s
python codes/A_parityplot/8_7_histogram/plot_mlip_dft_errors.py --model polar1m
python codes/A_parityplot/8_7_histogram/plot_mlip_dft_errors.py --model off
```

The script expects MLIP prediction CSVs in:

```text
codes/A_parityplot/8_6b_mlippredout/<model>/
```

Outputs go to:

```text
outputsfull/A_parityplot/8_7_histogram/<model>/
```

Four plots are written:

```text
<model>_absolute_force_error_hist.png
<model>_fractional_force_error_hist.png
<model>_absolute_energy_error_hist.png
<model>_fractional_energy_error_hist.png
```

The force plots use magnitudes. The energy plots are signed per-atom errors:
`(E_MLIP - E_DFT) / natoms` in eV/atom and the corresponding signed fractional
per-atom error.

Force histograms count atoms and use three colors:

- target isolated H atom
- nearest atom to the target isolated H atom
- all other atoms

Force plots are split into two stacked subplots with shared x-axis bins:
all other atoms on top, and target isolated H plus nearest atom on the bottom.

The target isolated H is identified as the H atom at the centered cluster
position `(12,12,12)`, not as atom index 0. This matches the cluster extraction
step, where the selected seed H is recentered in the 24 A vacuum box while atom
order remains sorted by original trajectory atom index.

Energy histograms count frames. By default energies are atom-reference-subtracted
using `codes/7_7b_clustervalidation/atomizationenergies.txt`; pass
`--no-reference` to compare raw total energies.
