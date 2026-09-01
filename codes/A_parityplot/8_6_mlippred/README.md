# 8_6 MLIP Single-Point Predictions

Run from the repository root:

```bash
python codes/A_parityplot/8_6_mlippred/predict_singlepoints.py --model polar1s --frames 0,180 --force
python codes/A_parityplot/8_6_mlippred/predict_singlepoints.py --model polar1m --frames 0,180 --force
python codes/A_parityplot/8_6_mlippred/predict_singlepoints.py --model off --frames 0,180 --force
```

Use CPU for a quick environment check:

```bash
python codes/A_parityplot/8_6_mlippred/predict_singlepoints.py --model polar1s --frames 0,2 --device cpu --force
```

Outputs go to:

```text
outputsfull/A_parityplot/8_6_mlippred/<model>/
```

Each run writes a frame-level `*_singlepoints.csv`, atom/force-level `*_forces.csv`,
and a predicted `*.xyz` trajectory unless `--no-extxyz` is set. Formal charges
are hard-coded as `H=+1`, `N=-3`, `O=-2`, and `S=-2`.

For MACE-POLAR, the system charge is computed separately for each frame from
that formal charge sum. This matches the ORCA input generation in `startup.py`
instead of incorrectly using charge `0` for every frame. The applied MLIP charge
is written to `mlip_charge_setting_e` in the summary CSV and the extxyz metadata.
Use `--charge <int>` only when intentionally overriding every frame to one fixed
charge for a diagnostic run.

Partial/predicted charge outputs are deprecated and are not written by this
active workflow.
