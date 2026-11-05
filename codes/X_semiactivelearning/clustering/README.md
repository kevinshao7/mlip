# Clustering tasks

- `extract_partial_clusters.py`: snapshot/review/extract from unfinished MD.
  The hash-frozen implementation remains in `tools/` for historical snapshots.
- `package_clusters.py`: actual implementation for packaging audited geometries
  and exporting a compact Git release. Its old `tools/` path is a compatibility import.

Use `partial_clusters_active.json` for the existing 2000-cluster release;
no new extraction is needed for CPU DFT. Shared geometry/species checks remain
in `production/`. See `../PARTIAL_CLUSTER_EXTRACTION.md`.
