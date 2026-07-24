# MLIP / ORCA Solubility Workflows

This repository contains scripts and local dependencies for NH3/H2S-in-water simulation workflows using molecular dynamics, ORCA DFT, and MACE-POLAR.

## Current Focus

The active cluster-validation workflow is:

1. Extract representative clusters from `outputsfull/r09_hot_w7n1`.
2. Generate ORCA input files from those clusters.
3. Run ORCA single-point calculations.
4. Compare MACE-POLAR cluster energies against ORCA DFT energies after subtracting DFT atomic reference energies.

Relevant scripts:

```text
codes/7_7b_clustervalidation/extract_dft_sized_clusters.py
codes/7_7b_clustervalidation/extract_small_cutoff_clusters.py
codes/7_7b_clustervalidation/compare_mace_polar_orca_clusters.py
codes/7_7b_clustervalidation/compute_trajectory_rdf.py
codes/7_7b_clustervalidation/summarize_npt_block_errors.py
codes/7_13a_orcaclusterssmall/expand77c.py
```

## Energy Convention

ORCA total energies are read from `FINAL SINGLE POINT ENERGY` in Hartree and converted to eV.

MACE-POLAR energies are compared to:

```text
DFT relative energy = ORCA total energy in eV - sum(DFT atomic reference energies)
```

The atomic reference energies are stored in:

```text
codes/7_7b_clustervalidation/atomizationenergies.txt
```

## Local Dependencies

Expected local checkouts:

```text
mlip/mace/
mlip/aseMolec/
```

The recent Windows Python used for local checks is:

```text
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe
```

MACE-POLAR may download its model on first use. Cache it inside the repository output area, for example by setting:

```powershell
$env:XDG_CACHE_HOME="C:\Users\shaoq\Documents\Mainz\mlip\outputsfull\.cache"
```

## Generated Data

Use `outputsfull/` for generated trajectories, summaries, plots, model caches, and analysis outputs.

Keep source scripts under `codes/`. Avoid adding large trajectories, checkpoints, ORCA scratch files, or cached model files to git.

## Useful Checks

Compile a modified script:

```powershell
python -m py_compile .\codes\7_7b_clustervalidation\compare_mace_polar_orca_clusters.py
```

Run the small-cluster ORCA/MACE comparison:

```powershell
python .\codes\7_7b_clustervalidation\compare_mace_polar_orca_clusters.py
```

Run ORCA locally over all generated small-cluster inputs, sequentially with 8 threads each:

```powershell
$env:OMP_NUM_THREADS=8; $env:MKL_NUM_THREADS=8; $env:OPENBLAS_NUM_THREADS=8; Get-ChildItem .\codes\7_13a_orcaclusterssmall\expand\*.inp | Sort-Object Name | ForEach-Object { orca $_.FullName | Tee-Object -FilePath ($_.DirectoryName + "\" + $_.BaseName + ".out") }
```
