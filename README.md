# MLIP / ORCA Solubility Workflows

> [!IMPORTANT]
> **Foundation-model policy (2026-10-28):** `polar-1-s` is retired for new
> simulations and training. Use the large MACE-POLAR checkpoint `polar-1-l`.
> Older `polar-1-s` names and artifacts are retained only for reproducibility.

This repository contains scripts and local dependencies for NH3/H2S-in-water simulation workflows using molecular dynamics, ORCA DFT, and MACE-POLAR.

> [!IMPORTANT]
> **JAX-MD CLI update:** Workflows that import the external `jax_md_cli`
> package must use the **`dev` branch of `jax-md-cli`** for MACE-POLAR. The
> development branch should now work with `model_type = "mace_polar"`. The
> MACE-POLAR launchers fail immediately if the checkout is not on `dev`.

Update and verify the external checkout before running one of these workflows:

```bash
cd /path/to/jaxmd-cli
git switch dev
git pull
git branch --show-current   # must print: dev
```

## Where to run

- **ORCA DFT: Viper (here) or BlueHive, using Slurm.** Maintain separate
  site-specific `.slurm` scripts because modules, MPI setup, executable paths,
  filesystem roots, and resource directives differ. Preserve scientific inputs
  when moving sites.
- **MACE-POLAR: GPU execution.** Fine tuning should run from a Python `.py`
  launcher on **dungeon**, or a GPU `.slurm` wrapper on **DAIS** calling the
  same workflow launcher and retaining its experiment logging.

See the [execution matrix and site examples](codes/WORKFLOW_MAP.md#machine-and-launch-policy).
X provides `codes/X_semiactivelearning/finetune_dais.slurm` and
`md_dais.slurm`, reusing the working DAIS GPU environment and preserving
Slurm GPU visibility. The separate D2 wrapper is retained for its own studies. Windows/local examples
below are retained as historical reference.

## Current data protocol (2026-11-03)

The active workflow is **[X_semiactivelearning](codes/X_semiactivelearning/README.md)**:
polar-1-l MD → clusters → dispersion-free wB97M DFT → fine tuning → held-out
force error. Continue acquisition/training at the same pressure until force
component RMSE is **<20 meV/Angstrom**, then increase pressure by **5 GPa**.
Start at 10 GPa; keep historical runs separate in the new `wb97m_step5` namespace.

[DATA_PROTOCOL.md](DATA_PROTOCOL.md) maps the active workflow and supporting
C6 conversion labels, atomic references, validators, site launchers and MD tools.
wB97M-V and D4rev-inclusive labels are excluded. A direct-method ORCA pilot is
required before production DFT; no new simulation/training has been launched.
The previous X controller/launcher/guide are preserved under its `deprecated/`.

## Local Dependencies

Expected local checkouts:

```text
mlip/mace/
mlip/aseMolec/
```

The JAX workflows additionally require an external `jax-md-cli` checkout on
its `dev` branch, as noted above.

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
