# Optional tools

Run from the workflow directory with `python tools/<script>.py --help`.

- `build_mace_source.py`: explicit source-environment build/install utility;
  see `../DAIS_ENVIRONMENT.md`. Never needed for routine production launches.
- `compare_foundation.py`: historical model comparisons using explicitly selected
  datasets; its legacy datasets/reference conventions do not select active X data.
- `visualize_force_outliers.py`: visualizes the historical comparison results.
- `extract_partial_clusters.py`: extraction-only workflow for unfinished P10 MD;
  see `../PARTIAL_CLUSTER_EXTRACTION.md` for snapshots, review and audit.
- `package_clusters.py`: combine audited split files into one `.xyz`, retaining
  per-record roles, original source metadata and a checksum manifest.

The comparison/environment tools are optional; partial extraction reuses the
production pipeline without launching simulations. Outputs and
Matplotlib caches go under `outputsfull/`.
