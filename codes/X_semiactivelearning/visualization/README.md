# Visualization tasks

- `plot_p10_finetuning.py`: published P10 training validation curves and true
  energy/force-component parity for all 2000 audited labelled clusters. The local
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
