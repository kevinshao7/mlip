# AGENTS.md

## Project Overview

This project investigates the solubility of NH3 and H2S in superionic water, motivated by the mystery of excess free H2S clouds in Uranus's atmosphere. Under simple solar-abundance chemistry, excess NH3 should react with H2S to form NH4SH, so free atmospheric H2S suggests missing physics or chemistry.

The broad goal is to use molecular simulation, DFT, and machine-learned interatomic potentials (MLIPs) to estimate the free energy of mixing and solubility of NH3 and H2S in water under relevant conditions.

## Scientific Plan

1. Build practical experience with molecular dynamics and MLIPs.

   * Fine-tune MACE-POLAR using Bingqing Chen's water data.
   * The relevant external repository is `ab-initio-thermodynamics-of-water`.

2. Generate DFT data for aqueous NH3 and H2S mixtures.

   * Use these calculations to fine-tune an MLIP suitable for NH3/H2S/water mixtures.

3. Use alchemical free energy methods.

   * Gradually turn on water-solute interactions.
   * Estimate free energies of mixing and solubilities for NH3 and H2S.

Useful scientific reading is stored in the `reading/` folder.

## Current Project Status

As of Monday June 16:

* Work has been organized into one GitHub repository intended to be interpretable by a Codex agent.
* Naive fine-tuning of MACE-POLAR is working.
* Molecular dynamics simulations of water can be run.
* Current understanding: isolated atom energies are not available in the training data, and `E0 = "estimate"` appears better than `"average"` for fine-tuning.

## Immediate Next Steps

Prioritize the following:

1. Understand how to use autocorrelation to estimate statistical errors from MD trajectories.

   * Do not assume MD frames are independent.
   * Prefer block averaging or autocorrelation-based error estimates where appropriate.

2. Understand MACE-POLAR multihead fine-tuning.

   * Compare multihead fine-tuning with naive fine-tuning.
   * Avoid overwriting or degrading pretrained water behavior unless this is intentional.

3. Become familiar with DFT workflows for NH3, H2O, and H2S.

   * Relevant resources:

     * ORCA manual: Density Functional Theory section.
     * ORCA / FACCTs documentation.
     * OMol25 dataset.
   * Goal: be able to run DFT calculations for NH3, H2O, and H2S over the weekend.

4. Familiarize with the rest of the pipeline from DFT data generation to MLIP training and free energy calculation.

## External Dependencies

This project depends on the following external repositories, which should be cloned separately into the project root:

```bash
git clone https://github.com/ACEsuit/mace.git
git clone https://github.com/imagdau/aseMolec.git
```
Visit github websites for relevant documentation

python environment is source ~/env/bin/activate
Expected layout:

```text
project-root/
  AGENTS.md
  README.md
  mace/
  aseMolec/
  reading/
  ...
```

## Coding Guidelines

* Prefer clear, explicit scientific code over clever abstractions.
* Use small, testable functions.
* Add comments for physical assumptions, units, and ensemble choices.
* Be explicit about units in variable names or comments.
* Avoid hard-coded absolute paths.
* Do not silently change simulation settings such as temperature, pressure, timestep, ensemble, periodic boundary conditions, or cutoffs.
* When modifying training or MD scripts, preserve reproducibility: record random seeds, input files, model checkpoints, and command-line arguments where possible.

## Molecular Simulation Guidelines

* Use ASE `Atoms` objects where appropriate.
* Be careful with:

  * periodic boundary conditions,
  * cell definitions,
  * units,
  * temperature and pressure conventions,
  * timestep stability,
  * thermostat/barostat choices,
  * whether the simulation is NVE, NVT, or NPT.
* For production MD analysis, account for autocorrelation.
* Do not report naive standard errors over frames unless clearly labelled as naive and likely over-optimistic.
* For correlated trajectories, prefer:

  * integrated autocorrelation time estimates,
  * block averaging,
  * or another justified correlated-sampling error estimate.

## MLIP / MACE-POLAR Guidelines

* Before changing training code, inspect the existing MACE and MACE-POLAR conventions.
* Be careful with energy references, especially isolated atom energies.
* Current note: for the water fine-tuning data, isolated atom energies appear unavailable, and `E0 = "estimate"` may be preferable to `"average"`.
* When fine-tuning:

  * distinguish naive fine-tuning from multihead fine-tuning,
  * avoid catastrophic forgetting if preserving pretrained capabilities matters,
  * save checkpoints and logs,
  * record exact training commands.
example codes/tutorials here:
https://colab.research.google.com/drive/1oCSVfMhWrqHTeHbKgUSQN9hTKxLzoNyb
https://colab.research.google.com/drive/1AlfjQETV_jZ0JQnV5M3FGwAM2SGCl2aU
https://colab.research.google.com/drive/1ZrTuTvavXiCxTFyjBV4GqlARxgFwYAtX#scrollTo=L7l0qtOVw9cz

## DFT Guidelines

* Treat DFT data generation as part of a reproducible pipeline.
* Preserve input files, output files, geometries, functional/basis settings, charge, multiplicity, and convergence settings.
* For ORCA calculations, check the ORCA manual and FACCTs documentation rather than guessing syntax.
* For NH3, H2O, and H2S calculations, verify molecular charge and spin multiplicity before running.

## Testing and Validation

Before finishing code changes, run relevant checks when available:

```bash
pytest
python -m compileall .
```

For scripts, prefer running a small smoke test with minimal data.

For MD or ML workflows, a useful smoke test should check that:

* the script starts correctly,
* input structures load,
* model checkpoints load,
* one or a few MD/training steps can run,
* outputs are written to the expected location.

## Git Guidelines

* Do not commit unless explicitly asked.
* Keep changes focused and easy to review.
* Explain what changed and how it was tested.
* Do not add large generated files, trajectories, model checkpoints, or DFT outputs to git unless explicitly requested.
* Prefer adding large-output directories to `.gitignore`.

## How Codex Should Help

When asked to modify the project:

1. First inspect the relevant files.
2. Identify the intended workflow before editing.
3. Make the smallest useful change.
4. Preserve scientific reproducibility.
5. Report:

   * files changed,
   * commands run,
   * tests or smoke tests performed,
   * any assumptions or unresolved issues.

When uncertain about scientific details, state the uncertainty rather than inventing a confident answer.
Always feel free to ask clarifying questions!