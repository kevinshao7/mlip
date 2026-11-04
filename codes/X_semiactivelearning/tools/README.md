# Optional tools

Run from the workflow directory with `python tools/<script>.py --help`.

- `build_mace_source.py`: explicit source-environment build/install utility;
  see `../DAIS_ENVIRONMENT.md`. Never needed for routine production launches.
- `compare_foundation.py`: historical model comparisons using explicitly selected
  datasets; its legacy datasets/reference conventions do not select active X data.
- `visualize_force_outliers.py`: visualizes the historical comparison results.

These tools are independent of the active P10 production pipeline. Outputs and
Matplotlib caches go under `outputsfull/`.
