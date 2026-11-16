# Visualization tasks

## Shared standard and swappable sources

**Remake all three models' plots and five outlier GIFs each with one command:**

```bash
python visualization/remake_parity.py
```

This automatically finds the latest complete saved s/m/l evaluation, prints its
path, and updates plots in that same evaluation folder. Predictions, plots,
outlier JSON and `outlier_gifs/` stay together under each model's s/m/l folder.
Prior plot versions are preserved under `plot_history/` inside that folder.
Finished matching GIFs from the old separate `energy_outliers_*` folders are
copied into the evaluation folder and reused. The deprecated separate
`energy_outliers_20261116T175919824655Z` release was removed after verifying its
GIFs and cluster records were retained in the unified evaluation folder.
Missing GIFs are generated only when needed. CPU only;
no model inference or DFT. Optional `--evaluation DIRECTORY` selects a saved
evaluation root explicitly.

Both error axes now use **logarithmic scales**. Energy x is
1000×DFT total energy/N; y is 1000×|MLIP−DFT total energy|/N in **meV/atom**.
Force y remains the vector-error norm in meV/Angstrom. The x axes remain linear.
Energy points are colored by cluster formal charge, recomputed from saved atom
symbols as H−3N−2O in the exact saved cluster order. The legend lists each charge
in elementary-charge units and its cluster count. Negative charges are blue,
positive charges red, with a symmetric scale about zero. Held-out structures,
when present, also have black crosses. Force colors remain H/N/O colors.
Charge counts are saved in metrics and each energy outlier records its charge.
Exact zero errors are omitted from logarithmic plots and counted in the figure;
they remain included in RMSE/MAE. Signed energy errors remain in saved data and
outlier JSON. The five largest
absolute per-atom energy residuals are circled and labelled `rank: #cluster_index`
(zero-based saved prediction order), with the same ranking in `energy_outliers.json`.
Force units and definitions are unchanged. This supersedes historical signed/total-energy
plot conventions below; saved predictions remain in eV and eV/Angstrom.

To redraw one model and render its five worst energy clusters from saved data:

```bash
python visualization/plot_parity.py --from-evaluation /path/to/evaluation/m
python visualization/visualize_saved_outliers.py /path/to/evaluation/m --output /new/path/to/m_outliers
```

The GIF adapter reuses `visualize_force_outliers.py`, verifies dataset hashes
and exact reference-energy/force/count/symbol/split order, and never loads a model.
It shows the whole nonperiodic cluster. Gold rings mark the largest force-error
atom by default (`--highlight-atoms 3` selects three); atom indices are shown,
with DFT/MLIP/error arrows on the worst force-error atom. These highlights are
force diagnostics, not atomwise attribution of cluster energy error. Each output
includes a GIF, extxyz with saved MLIP forces and atom errors, and `outliers.json`
with original source/frame/centre identity and prediction hash. Defaults are
60 rotation frames at10fps. Existing files are never overwritten.

All future energy/force parity plots must use `parity_standard.py`, shared by
`plot_parity.py` and the historical `plot_p10_finetuning.py`. Change the data-source
manifest rather than copying or changing the renderer for a new model/dataset.
The established plots are energy residual and force vector-error plots, with
the conventions described below; compatibility filenames contain `parity`.

`parity_sources.json` pins the latest completed local fine-tuned POLAR-1-S/M/L
foundation/SWA runs (November 13–14): stage-two exported models, float64, and
the same frozen 1000-structure validation split. Validation is descriptive,
not an independent test estimate. The 50 held-out evaluation structures are
excluded. No new model or training experiment is selected by these plots.

Run these yourself from the X workflow directory on your allocated GPU:

```bash
# Evaluate all three models and save predictions plus plots
python visualization/plot_parity.py --device cuda
# Alternatively, save evaluations first without rendering any plots
python visualization/plot_parity.py --device cuda --evaluate-only
# Redraw a saved model evaluation without GPU, MACE loading or inference
python visualization/plot_parity.py --from-evaluation /path/to/evaluation/m
```

DFT ground truth is read from the existing audited labels; no DFT calculations
are rerun. Each model's `predictions.npz` saves DFT and MLIP energies (eV),
DFT and MLIP forces (eV/Angstrom), atom counts, split roles and atom symbols.
It and provenance in `metrics.json` are written **before** plotting, so a plot
failure or later style change needs no repeat inference. `--evaluate-only`
writes these same reusable files; numerical plot metrics are added on redraw.
Outputs use a fresh dated directory under
`outputsfull/X_semiactivelearning/visualization`, with `s/`, `m/`, `l/` folders
and a source-manifest copy. Existing outputs are preserved.

Use `--models m` to select one source or `--sources /path/to/sources.json`
to swap sources. Each manifest source has `label`, `model` (`path`, `sha256`),
`dtype`, and `datasets` (each with `path`, `sha256`, `count`, `role`). Paths
are repository-relative or absolute. Supported labels are audited nonperiodic
H/N/O absolute dispersion-free wB97M/def2-TZVPD `REF_energy`/`REF_forces`,
with verified formal charge and singlet metadata. A different chemistry or
reference convention requires an explicitly validated adapter. No E0 subtraction.
`--check` verifies source hashes, counts and reference provenance on CPU without
inference or rendering. Split metrics are per-atom energy RMSE and force-component
RMSE; figure annotations retain total-energy and force-vector RMSE/MAE.

## Historical entry points

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
