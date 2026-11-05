# Visualization tasks

- `campaign_dashboard.py`: read-only campaign dashboard entry point.
- `visualize_force_outliers.py`: force-outlier GIF implementation, with a
  compatibility import at its previous `tools/` path. Select its dataset
  explicitly; its historical defaults do not select the new unlabelled release.

Use `python visualization/<script>.py --help` from the workflow directory.
Existing campaign plotting functions remain shared in `production/campaign.py`;
this organization preserves frozen MD source hashes and restart compatibility.
