# 8_8 MLIP Target-H Breakdown

Run from the repository root:

```bash
python codes/A_parityplot/8_8_breakdown/plot_mace_off_target_h_breakdown.py
```

By default this processes all three models:

```text
off, polar1s, polar1m
```

To run one model:

```bash
python codes/A_parityplot/8_8_breakdown/plot_mace_off_target_h_breakdown.py --model polar1s
```

The script reads:

```text
codes/A_parityplot/8_6b_mlippredout2/<model>/*_forces.csv
codes/A_parityplot/8_6b_mlippredout2/<model>/*_singlepoints.csv
outputsfull/A_parityplot/8_5_bluehiveDFT/*.out
```

It identifies the isolated H as the H atom at `(12,12,12)`, finds the nearest
other atom, and uses blue points when the nearest atom is oxygen and red points
when the nearest atom is hydrogen.

The x axis is always isolated-H to nearest-atom separation in Angstrom.

For each model, the script writes:

```text
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_target_h_breakdown.csv
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_target_h_longitudinal_force_error_abs.png
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_target_h_longitudinal_force_error_fractional.png
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_target_h_perpendicular_force_error_abs.png
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_target_h_perpendicular_force_error_fractional.png
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_total_energy_error_signed.png
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_total_energy_error_fractional.png
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_per_atom_energy_error_signed.png
outputsfull/A_parityplot/8_8_breakdown/<model>/<model>_per_atom_energy_error_fractional.png
```

Force errors are decomposed along the isolated-H to nearest-atom vector. The
longitudinal force error plots are signed projected errors. Positive means
`F_MLIP - F_DFT` points from the isolated H toward its nearest atom, so MLIP is
more attractive than the DFT baseline and the DFT baseline is more repulsive
along that isolated-H to nearest-atom axis. The perpendicular force plots remain
magnitudes because there is no unique scalar sign for the perpendicular vector
component. Energy errors are signed as `E_MLIP - E_DFT`; by default they are
atom-reference subtracted using
`codes/7_7b_clustervalidation/atomizationenergies.txt`.

## Charge-Colored Breakdown Diagnostic

To make the same eight target-H breakdown plots, but color points by total
system charge instead of nearest atom type:

```bash
python codes/A_parityplot/8_8_breakdown/plot_charge_combo_energy_diagnostic.py --model polar1m
```

This uses `formal_charge_sum_e` only. Partial/predicted charge sources are
deprecated and intentionally unavailable:

```bash
python codes/A_parityplot/8_8_breakdown/plot_charge_combo_energy_diagnostic.py --model polar1m --charge-source formal
```

Outputs:

```text
outputsfull/A_parityplot/8_8_breakdown/<model>/charge_colored/<model>_target_h_breakdown_by_total_charge.csv
outputsfull/A_parityplot/8_8_breakdown/<model>/charge_colored/<model>_*.png
```
