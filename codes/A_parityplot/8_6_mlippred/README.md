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
are hard-coded as `O=-2` and `H=+1`.
