# Visualization tasks

- `plot_p10_finetuning.py`: published P10 training validation curves and energy/force residual plots for all 2000 audited labelled clusters. The local
  run timed out after epoch 195; this script uses the hash-pinned epoch-9 export,
  not a completed 500-epoch model. It writes a new dated directory under
  `outputsfull/X_semiactivelearning/visualization/`, including curves CSV,
  predictions NPZ, PNG plots and split-specific metrics JSON. No E0 subtraction.
  Acquisition and the 50 held-out evaluation clusters are identified separately;
  combined parity is descriptive, not independent test performance.
  From this workflow directory run
  `python visualization/plot_p10_finetuning.py --device cuda` on an allocated GPU,
  or use `--curves-only` to plot logged validation performance without inference.
  The script evaluates the held-out set only when run without `--curves-only`.
  Existing `tools/compare_foundation.py` targets historical P31 data by default
  and its force plot is error versus magnitude rather than component parity.

- `campaign_dashboard.py`: read-only campaign dashboard entry point.
- `visualize_force_outliers.py`: force-outlier GIF implementation, with a
  compatibility import at its previous `tools/` path. Select its dataset
  explicitly; its historical defaults do not select the new unlabelled release.

Use `python visualization/<script>.py --help` from the workflow directory.
Existing campaign plotting functions remain shared in `production/campaign.py`;
this organization preserves frozen MD source hashes and restart compatibility.

Regenerate plots after editing `plot_history` or `plot_parity`, without inference:

```bash
python visualization/plot_p10_finetuning.py --from-evaluation /path/to/completed/evaluation
```

Wait for the running evaluation to finish: the source directory must contain
`predictions.npz`, `metrics.json` and `validation_history.csv`. This mode does
not import MACE or load the model/dataset. Plots and recomputed metrics go into
a fresh dated directory; optionally select a new directory with `--output`.
The original evaluation is preserved.

Select parity predictions on the same 2000 labelled clusters:

```bash
python visualization/plot_p10_finetuning.py --model found --device cuda
python visualization/plot_p10_finetuning.py --model finet --device cuda
```

`found` uses the cached, hash-verified polar-1-l checkpoint bound to the dataset's
original MD source plan, resolved through `partial_clusters_active.json`.
`finet` (the default) uses the pinned epoch-9 export. Each evaluation saves its
own predictions and metrics with model identity in a separate dated directory.
Foundation mode creates energy/force parity plots; training curves apply only
to fine tuning. `--from-evaluation DIRECTORY` infers the saved model identity,
including older epoch-9 results, without inference. An explicit conflicting
`--model` is rejected. Foundation results need no validation-history CSV.

Plot conventions: energy x = absolute DFT energy (meV), y = MLIP − DFT
energy (meV), without per-atom normalization. Force x = ‖F_DFT‖ and
 y = ‖F_MLIP − F_DFT‖, both in meV/Å, with H blue, N green and O red.
The force error is the norm of the vector difference. Plot filenames remain
`energy_parity.png` and `force_parity.png` for compatibility. Numerical split
metrics retain their existing per-atom energy and force-component definitions.
New predictions archives include `atom_symbols`. Replotting old archives recovers
symbols from the hash-verified labelled dataset and requires exact reference
energy, force, count and split-order agreement. Replot output includes an enriched
predictions archive so subsequent redraws need no source dataset.

Both residual figures display RMSE and MAE for the combined set. Energy
annotations use total cluster energy errors in meV. Force annotations use vector
errors: RMSE = sqrt(mean(||F_MLIP − F_DFT||²)), MAE = mean(||F_MLIP − F_DFT||),
in meV/Å, overall and separately for H/N/O. These vector statistics differ from
the force-component RMSE retained in `metrics.json`.

Verified residual figures and self-contained saved predictions (2000 clusters):
- Foundation: `outputsfull/X_semiactivelearning/visualization/p10_saved_20261110T181841275108Z`
- Fine-tuned: `outputsfull/X_semiactivelearning/visualization/p10_saved_20261110T181857041077Z`

Paths are relative to the repository root. Use either with `--from-evaluation`.
