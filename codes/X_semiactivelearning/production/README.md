# Shared production code

Task entry points and standalone tools are organized in `../md/`,
`../clustering/`, `../dft/` and `../visualization/`. The published-cluster CPU
pipeline is `../dft/viper_cpu.py`; its `../submitcpu.sh` submits 200 separate
jobs only when explicitly invoked.

These shared modules serve all pressures. Pressure/site/run names belong in generated
launch bundles and data folders, not duplicate source trees.

| Pipeline stage | Main implementation |
| --- | --- |
| GPU MD and restart | `p10_batch.py`, `md_segments.py`, `campaign.py:command_md` |
| Intermediate cut clusters | `workflow_species.py`, `cluster_quality.py`, `campaign.py:command_clusters` |
| CPU DFT preparation/validation | `site_dft.py`, `method_pilot.py`, `orca_output_validation.py` |
| GPU training | `launch_single_gpu.py`, `training_bridge.py` |
| Shared policy/controller/reporting | `workflow_policy.py`, `autonomous.py`, `campaign_dashboard.py` |

Use the four top-level command adapters for production entry points. Optional
export/analysis tools live in `../tools/`; isolated checks live in `../tests/`.
The current dataset pointer is `../partial_clusters_active.json`. See
`../PARTIAL_CLUSTER_EXTRACTION.md` for the pressure/stage data layout.
