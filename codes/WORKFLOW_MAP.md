# Workflow map

This repository contains several dated studies.  Directory names encode their
creation history, not whether they are the current workflow.  Use this map
before selecting code or training data.

| Area | Purpose | Status | Dataset relationship |
| --- | --- | --- | --- |
| `B_conditionsproduction`, `B2_conditionsfine` | Production MD and condition-specific job generation | Active simulation support | Produces trajectory data; it is not a training split. |
| `C_DFTproduction` | Original water-cluster extraction and ORCA generation | Provenance source | Its 162 water-only DFT structures are retained in D2 for provenance. |
| `C3_DFTproductionstopH2` | Nitrogen/closest-approach cluster extraction and ORCA generation | Active DFT source | Supplies the newer nitrogen-bearing configurations in the D2 target dataset. |
| `C2_atomizationDFT` | Isolated-atom ORCA references | Shared reference workflow | Regenerates the H/N/O/S reference-energy CSV. |
| `D2_naive` | Target-only PolarMACE fine tuning | **Current target fine-tuning workflow** | `data/target_all.xyz` combines the formal-charge-validated C_DFT and C3 DFT data. |
| `D_MHFT` | Multihead fine tuning with OMol replay | Separate legacy/experimental workflow | Its 162-frame target data is oxygen-only and must not replace D2 data for N-bearing fine tuning. |
| `7_29_finetunepolar`, `A_parityplot`, `7_7b_clustervalidation` | Earlier H2/path validation and diagnostic studies | Historical/validation support | Not the source of the D2 training split. |
| `old`, `archive` | Superseded scripts and snapshots | Preserve until an approved cleanup | Never use as a default workflow. |

## D2 dataset catalog

| File | Meaning | Use in current workflow |
| --- | --- | --- |
| `data/target_all.xyz` | 227 formal-charge-validated H/N/O ORCA DFT frames, including 45 N-bearing frames | Authoritative source; never train directly without a declared split. |
| `data/target_train.xyz`, `target_valid.xyz`, `target_test.xyz` | Seeded 80/10/10 source-family/nitrogen-presence split derived from `target_all.xyz` | Current train/validation/test inputs; audit before a new experiment. |
| `data/target_all_2026-09-03_ON_closest_approach.xyz` | Earlier dated ON closest-approach snapshot | Historical baseline; not the current source. |
| `data/archive/water_only_delta_experiments_2026-09-29/` | Water-only residual/delta experiments | Archived; not an input to the current N-bearing naive workflow. |

Run `python analyze_dataset_distribution.py` from `D2_naive` to make the
split composition, formal-charge checks, duplicate checks, and leakage checks
explicit.  Generated reports are written below `data/dataset_distribution/`.

No file is deleted merely because it is old: deletion needs an approved,
enumerated cleanup list after checking references and provenance value.
