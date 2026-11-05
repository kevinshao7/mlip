# Dispersion-free wB97M labels

The user selected this D4rev calculation plus explicit energy/force subtraction
path for future X/P10 labels on 2026-11-04. See
[X/P10 provenance and deployment status](../X_semiactivelearning/DFT_PROVENANCE.md).
This decision does not modify this historical release or certify a new batch.

## Scientific significance and released dataset

This release separates the learnable wB97M electronic energy/force surface from
the explicitly evaluated D4rev correction. Training on these targets allows
dispersion to be added once at inference, with its geometry, charge, and
periodicity dependence evaluated by DFT-D4. Subtracting only the energy would
leave inconsistent force targets; this workflow removes both terms using the
values printed by the original ORCA calculation. It does not remove all
electronic correlation or turn wB97M into a strictly short-range functional.
Improved accuracy or transferability is a hypothesis requiring a controlled
parent/child training comparison, not an established result of this conversion.

The released conversion is the user-run snapshot
[`20261103T171628_022947Z`](../../outputsfull/acquisition/wb97m_nodisp/20261103T171628_022947Z/):
**363 source outputs, 266 accepted configurations, 97 excluded outputs without
normal termination**. All accepted configurations passed charge, multiplicity,
method, gradient-order, and reconstruction checks. This is a label-validity
result, not a certification of sampling coverage or independent samples.

The release includes:

- [`wb97m.extxyz`](../../outputsfull/acquisition/wb97m_nodisp/20261103T171628_022947Z/wb97m.extxyz): dispersion-free targets plus total/D4 audit labels.
- [`report.json`](../../outputsfull/acquisition/wb97m_nodisp/20261103T171628_022947Z/report.json): accepted frame mapping and exclusions from the actual user invocation.
- [`provenance.json`](../../outputsfull/acquisition/wb97m_nodisp/20261103T171628_022947Z/provenance.json): portable repository-relative paths and SHA256 hashes of source files, released artifacts, and conversion code.
- [Raw acquisition outputs](../../outputsfull/acquisition/outputs/): all 363 preserved stdout files, including failures. Echoed ORCA inputs retain the original method and geometry declarations. Original standalone input/property/scratch files are not included in this release.

Historical absolute Windows paths in extxyz/report metadata identify the
original local invocation; use repository-relative paths in provenance.json
after moving the repository. No values in the user's snapshot were rewritten
for publication. Original source-job paths echoed in stdout identify the HPC
acquisition launch, but the release makes no claim of reconstructing missing
Slurm or scratch artifacts. The earlier developer smoke-test snapshot is not
the published dataset.

## Generate another conversion

From the repository root, run:

```sh
python codes/C6_wb97m_nodisp/prepare_data.py
```

Requires NumPy and ASE. Defaults to `outputsfull/acquisition/outputs/*.out`.
Every invocation creates a new UTC timestamped directory below
`outputsfull/acquisition/wb97m_nodisp/`, containing `wb97m.extxyz` and
`report.json` with accepted-frame indices, SHA256 provenance, and rejected files.
Use `--input-dir` or `--output-dir` to override; an existing output directory is
never reused. No raw ORCA data or existing training files are changed.

The ORCA-printed input is the geometry/charge/spin provenance, including when
the original .inp is unavailable. This converter only accepts one completed
`wB97M-D4rev def2-TZVPD EnGrad RIJCOSX` single-point job per file and H/N/O
clusters. Formal charge is recomputed as H - 3N - 2O and must match both echoed
input and computed-output total charge. Multiplicity must match input/output
and electron-count parity. Unsupported elements, missing gradients, atom-order
mismatches, failed jobs, or ambiguous multiple-job outputs are rejected.

The targets are:

```text
REF_energy = (E_total - E_D4) * Hartree_to_eV
REF_forces = -(gradient_total - gradient_D4) * Hartree_to_eV / Bohr_to_Angstrom
```

Both gradients are in Hartree/Bohr. Energies are absolute electronic energies
in eV, forces in eV/Angstrom. No atomization reference is subtracted.
The original total and removed D4 labels are also retained as `total_energy`,
`total_forces`, `D4_energy`, `D4_forces`. Re-adding D4 must reconstruct total
labels, checked before writing and after reading back. Printed ORCA precision
limits the subtraction (roughly 1e-8 eV energy, 1e-7 eV/Angstrom forces);
extxyz force arrays have eight decimal places.

These are isolated, nonperiodic clusters with no cell or real-space cutoff,
and no thermodynamic ensemble; parent MD conditions are not altered. Do not
interpret the extracted geometries as newly equilibrated samples.

For MACE training use energy key `REF_energy` and forces key `REF_forces`.
Preserve the existing train/validation/test assignment by source identity; this
script deliberately does not resplit data. Validation/test targets must also
have D4 removed. Do not mix these labels with D4-inclusive labels. Neutral-atom
E0s must be consistent with this underlying wB97M method and basis: isolated
atomic pair/triple dispersion vanishes, but verify that fact in the reference
outputs before reusing existing C5 E0s. Never use old-functional references.
No launcher defaults or approved training experiment are changed here.

## Add D4 at inference

`d4_inference.py` exposes `with_d4(ml_calculator, charge=explicit_total_charge)`
as an ASE sum calculator, adding energy, forces, and periodic stress. Use it
with a model actually trained on the dispersion-free labels. The helper passes
total charge explicitly to DFT-D4; it does not infer it from partial charges.
Its D4rev parameters are s6=1, s8=0.842, a1=0.359, a2=4.668, s9=1, alp=16.
It uses D4 (EEQ), including ATM; library-default cutoffs apply. The actual
inference cell/PBC are passed through, so periodic inference includes periodic
dispersion even though training clusters were nonperiodic.

Install the optional `dftd4` Python library in the inference environment, then
validate that backend against the ORCA corrections before using it:

```sh
python codes/C6_wb97m_nodisp/d4_inference.py path/to/wb97m.extxyz
```

The check fails if energy or force-component differences exceed 1e-6 eV per
cluster or 1e-6 eV/Angstrom respectively. Resolve version, model, or cutoff
differences rather than loosening tolerances without evidence. The source
outputs report DFTD4 V3.4.0. Backend equivalence and periodic deployment have
not been validated locally because `dftd4` is not installed.

Sources: [ORCA dispersion corrections and D4rev parameters](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/dispersioncorrections.html),
[DFT-D4 Python API](https://dftd4.readthedocs.io/en/latest/reference/python.html).

## Validation

```sh
python -m unittest discover -s codes/C6_wb97m_nodisp -p test_prepare_data.py
```

Tests check known real-output energy/force subtraction, extxyz reconstruction,
charge/spin rejection, missing D4 gradient rejection, and failed-run rejection.
