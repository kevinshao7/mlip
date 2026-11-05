# AGENTS.md

## Project

This repository supports molecular simulation, ORCA DFT, and MACE-POLAR workflows for NH3/H2S solubility in superionic water, motivated by Uranus atmospheric chemistry.

Use the local code and generated data as the source of truth. The project has moved beyond initial onboarding; avoid reviving old June planning notes unless they are directly relevant.

## Start here for current work

The active workflow is **`codes/X_semiactivelearning/`**. Read its `AGENTS.md`
first, then only the relevant section of `DEBUGGING.md` or `P10_BATCH.md`.
The prepared P10 ensemble is selected by `p10_batch_active.json`; resolve that
pointer instead of choosing the newest-looking directory or a deprecated plan.
The generic pressure controller (`autonomous.py`, default `wb97m_step5`) and the
named P10 batch (`p10_batch.py`) are different entry points/output namespaces.

Current user scope is **preparation only**: no job submission or production
MD/DFT/training without a later explicit launch instruction. Ten GPU replicas
must be separate one-GPU jobs, each with its own restart dependencies. Consult
the active plan for physical settings; generic defaults are not batch overrides.

Keep routine work narrow: inspect the relevant generator/function, make the
requested change, run its focused check once, update the lab notebook, and
report the result. Do not repeat onboarding, scan all generated data, download
an already-pinned model again, or redesign adjacent workflows for a small edit.
Report a separate scientific defect with evidence before expanding scope.
For documentation-only edits, link/path checks and `git diff --check` suffice;
do not regenerate launch bundles or rerun simulations/unit suites.

## Layout

```text
codes/                  workflow source and supporting dated studies
codes/X_semiactivelearning/           primary active workflow
codes/C6_wb97m_nodisp/                supporting label/pilot evidence
codes/7_7b_clustervalidation/         historical comparison tools
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

On Viper, use `/viper/ptmp1/kshao/mlip/.venv/bin/python` (Python 3.11.11).
MACE is installed editable from the official GitHub develop checkout, version
0.3.18, commit `7215fadd3b591444ab2c07634e84419685a48e87`. This supersedes
DAIS-only paths when working on Viper. See
[Viper environment](codes/X_semiactivelearning/VIPER_ENVIRONMENT.md).
The site-specific DAIS record below remains unchanged for that host.

On the currently verified DAIS checkout, use:

```text
/dais/fs/scratch/kshao/mlipenv/bin/python
```

It is Python 3.11 with ASE/NumPy/Torch/MACE available. Bare `python` was absent;
`/usr/bin/python3` was Python 3.6 and cannot compile this workflow. Verify the
interpreter once per host, then reuse it. The Windows Python mentioned in old
notes is historical context, not the default here.

`mace/` is the GitHub ACEsuit/mace v0.3.17 checkout, installed editable into
mlipenv on 2026-11-04 without pip. Imports resolve to `mace/mace/__init__.py`.
`aseMolec/` remains a checkout placeholder. Exact versions, source-build and
rollback records: [DAIS environment](codes/X_semiactivelearning/DAIS_ENVIRONMENT.md).
Keep model caches under `outputsfull/.cache` through `XDG_CACHE_HOME`.

## Machines and launch policy

**ORCA DFT must be runnable on Viper or BlueHive.** Determine the actual host
from the environment; do not infer it from an old README saying "here". Both use
Slurm, but require separate site-specific `.slurm` templates or explicit site
profiles. Adapt module/MPI setup, ORCA executable paths, filesystem roots,
partitions, resource directives, and log paths for each site. Moving a
calculation between sites must preserve its geometry, functional, basis,
explicitly verified formal charge, and spin multiplicity.

**MACE-POLAR simulations and fine tuning run on GPUs.** Fine tuning must
support a direct Python `.py` workflow launch on **dungeon** and a GPU
`.slurm` submission on **DAIS**. The DAIS wrapper must call the workflow's
Python launcher, retaining the same training defaults, unique run directories,
and workflow-level `hyperparameters.txt`. Preserve Slurm's assigned GPU
visibility; do not overwrite it with a dungeon physical GPU ID. Reuse the
working DAIS GPU environment from
`codes/B_conditionsproduction/NPTMACEproduction_base.slurm` (`gcc/14`,
`cuda/12.8`, `/dais/fs/scratch/kshao/mlipenv`).

Windows commands and dungeon-local DFT instructions in older notes are
historical or workflow-specific context, not the default deployment policy.
See `codes/WORKFLOW_MAP.md` for site examples and current implementation gaps.

## Scientific Rules

Be explicit about units, thermodynamic ensemble, cutoffs, periodic boundary conditions, charge, spin multiplicity, and reference energies.

Do not silently change simulation settings such as temperature, pressure, timestep, ensemble, cutoff length, model checkpoint, functional, basis, charge, or multiplicity.

For MD analysis, do not treat correlated frames as independent. Prefer block averaging, autocorrelation estimates, or clearly labelled exploratory statistics.

## Energy References

The active X/C6 wB97M-only labels use **absolute electronic `REF_energy` in eV**
and `REF_forces` in eV/Angstrom. Do not subtract legacy atomic references from
these labels. Matching E0s are explicitly audited/selected by the training path.
ORCA `FINAL SINGLE POINT ENERGY` is in Hartree; conversion is mandatory.

Older `7_7b_clustervalidation` atomization comparisons use a different convention:
ORCA total energy in eV minus the sum of their documented atomic references.
Use that convention only for that legacy comparison, never as a repository-wide
rule. Do not mix functional/dispersion reference methods or their E0 tables.

## ORCA Workflow

## CRITICAL DFT INVARIANT — FORMAL CHARGE IS MANDATORY

**Never guess, inherit, or silently trust a charge value when creating, rerunning,
converting, validating, or training on ORCA DFT data.** Every ORCA input must
use the molecule's explicitly calculated formal charge, and that same value
must be retained in the resulting extxyz `charge` metadata.

For the current H/N/O target dataset, the required project convention is:

```text
formal charge = (+1 × number of H) + (−3 × number of N) + (−2 × number of O)
```

Before an ORCA run, calculate and record this value in the input. Before data
conversion or MACE training, recompute it from the structure and require exact
agreement with both the ORCA-declared total charge and extxyz `charge` field.
**Reject mismatches from every training, validation, test, and evaluation
dataset; do not repair them by guessing or carry them forward.** Preserve the
raw ORCA files for auditability, but never train on charge-inconsistent data.

For a chemistry outside this H/N/O convention, define and document the
appropriate formal-charge rule before generating DFT inputs; fail validation if
no rule is available. Charge and spin multiplicity are independent and both
must be explicitly verified.

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

## Fine-Tuning Experiment Protocol

For every MACE fine-tuning iteration, use the workflow's launcher rather than
calling `run_train.py` directly. Each launch must create a new directory with
a unique datetime in its name and must append to the single workflow-level
`hyperparameters.txt` before optimization begins. Never reuse or overwrite a
prior run directory.

Treat that single file as the living lab notebook. Agents must append a dated
entry to the workflow-level `hyperparameters.txt` whenever they inspect,
diagnose, compare, or decide what to do next with a run. Do not replace its
launch records.
Each appended entry must state:

- the observation and exact evidence (metrics, epoch, log path, or plot);
- the current diagnosis or uncertainty;
- the proposed next test;
- why that test is the most informative next step; and
- the one parameter/factor it changes relative to this run.

Write an entry after an interrupted or failed run too. Before starting a child
run, append the planned one-factor change and rationale to the single log,
then record the corresponding `--changed-parameter` and `--change-note` in
the child launch. This preserves the causal chain between experiments.

**Keep code and experiment records synchronized.** Whenever an agent changes
fine-tuning code, defaults, data selection, or documentation, it must in the
same task append a dated explanation to the workflow's single
`hyperparameters.txt`. Conversely, a diagnosis or next-test decision made
in an experiment record must be reflected in the relevant workflow code or
README when it changes the default procedure. Never leave a code change or a
scientific decision recorded in only one place.

When preparing the next approved experiment, update the workflow defaults in
code as well as the experiment log. The user must be able to start that exact
prepared experiment with only `python launch_single_gpu.py` from the workflow
directory—no additional flags, keyword arguments, or remembered overrides.
Record the same active setting and rationale in `hyperparameters.txt` in the
same task. Do not tell the user to add flags for a change that the agent has
already prepared.

Each launch record must include exact locations for the run directory, training
log, checkpoint model, and compiled model (if compilation succeeds). Tell the
user the single workflow-level log path. A user can launch the current baseline
simply with `python launch_single_gpu.py` from its workflow directory; the
launcher appends the run record automatically.

Run experiments in this order:

1. Start from the documented validated baseline.
2. Change **exactly one** scientific or training factor.
3. Set `--changed-parameter` to that one factor and explain the rationale in
   `--change-note`.
4. Append the diagnosis and next-test rationale to the parent's
   `hyperparameters.txt`.
5. Compare the new validation metrics with its direct parent run before making
   another change.

Do not combine data-split, model, optimizer, learning-rate, loss-weight,
foundation-checkpoint, precision, or scheduler changes in a single trial.
Record a failed trial as faithfully as a successful one; it is part of the
experimental evidence.
