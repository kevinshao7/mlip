# AGENTS.md

## Project

This repository supports molecular simulation, ORCA DFT, and MACE-POLAR workflows for NH3/H2S solubility in superionic water, motivated by Uranus atmospheric chemistry.

Use the local code and generated data as the source of truth. The project has moved beyond initial onboarding; avoid reviving old June planning notes unless they are directly relevant.

## Layout

```text
codes/                  dated workflow scripts
codes/7_7b_clustervalidation/
  extract_dft_sized_clusters.py       extracts roughly 18-20 atom clusters
  extract_small_cutoff_clusters.py    extracts smaller cutoff-defined clusters
  compare_mace_polar_orca_clusters.py compares MACE-POLAR energies/forces to ORCA DFT outputs
  compute_trajectory_rdf.py           computes trajectory RDF plots
  summarize_npt_block_errors.py       estimates NPT block-average errors
codes/7_13a_orcaclusterssmall/
  clusters/             small cluster xyz files
  expand/               ORCA inputs and outputs for those clusters
outputsfull/            generated trajectories, analyses, plots, caches
mace/                   local ACEsuit MACE checkout
aseMolec/               local aseMolec checkout
```

Large generated files belong under `outputsfull/` or dated workflow folders, not in new source locations.

## Environment

The Windows Python used for recent local checks is:

```text
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe
```

The local MACE checkout is imported from `mlip/mace`. MACE-POLAR model downloads should cache inside `outputsfull/.cache` by setting `XDG_CACHE_HOME` rather than writing to the user home directory.

## Scientific Rules

Be explicit about units, thermodynamic ensemble, cutoffs, periodic boundary conditions, charge, spin multiplicity, and reference energies.

Do not silently change simulation settings such as temperature, pressure, timestep, ensemble, cutoff length, model checkpoint, functional, basis, charge, or multiplicity.

For MD analysis, do not treat correlated frames as independent. Prefer block averaging, autocorrelation estimates, or clearly labelled exploratory statistics.

## Energy References

MACE-POLAR cluster energies are compared to DFT cluster energies after subtracting DFT atomic reference energies from `codes/7_7b_clustervalidation/atomizationenergies.txt`.

ORCA `FINAL SINGLE POINT ENERGY` values are in Hartree and must be converted to eV before comparison.

For a cluster:

```text
DFT relative energy = ORCA total energy in eV - sum(DFT atomic reference energies)
```

Compare that value to the MACE-POLAR predicted cluster energy.

## ORCA Workflow

Preserve ORCA `.inp`, `.out`, `.property.txt`, and related generated files for reproducibility.

For ORCA outputs, verify both:

```text
FINAL SINGLE POINT ENERGY
ORCA TERMINATED NORMALLY
```

Generated Slurm output and ORCA stdout should go under `mlip/outputsfull`, not the source tree, unless the user explicitly asks for local interactive runs.

## Coding Guidelines

Prefer small, explicit functions and conservative edits. Add comments only for non-obvious physical assumptions, units, reference-energy conventions, or workflow traps.

Avoid hard-coded absolute paths in new Python code. Existing cluster/ORCA scripts may contain machine-specific HPC paths; keep them centralized in templates.

Do not commit unless explicitly asked. Do not delete or rewrite generated data unless the user asks or the workflow clearly regenerates that exact output.

## Validation

Before finishing code changes, run the narrowest useful check:

```bash
python -m py_compile path/to/script.py
```

For workflow scripts, prefer a small smoke test that confirms inputs load and outputs are written to the expected directory. Report any dependency, network, or model-cache blockers directly.
