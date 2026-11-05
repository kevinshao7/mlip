# P10 DFT label provenance and selected calculation path

User decision: 2026-11-04. Retain the established C6 calculation route:
`wB97M-D4rev def2-TZVPD EnGrad RIJCOSX`, ORCA 6.1.1 on Viper.
The training target remains dispersion-free wB97M/def2-TZVPD.

## Label derivation

For each fixed isolated geometry (PBC=False), recompute formal charge as
H - 3N - 2O and require agreement with input, computed output and extxyz.
Verify singlet multiplicity and even electron count separately.

```text
REF_energy = (E_total - E_D4) * Hartree_to_eV
REF_forces = -(gradient_total - gradient_D4) * Hartree_to_eV / Bohr_to_Angstrom
```

Use ORCA's FINAL SINGLE POINT ENERGY, Dispersion correction, CARTESIAN
GRADIENT and DISPERSION GRADIENT. Require normal termination, finite values,
exact atom ordering and one unambiguous job. Reject missing D4 gradients;
never subtract energy alone. No legacy atomic-energy subtraction applies.
Preserve total_energy, total_forces, D4_energy and D4_forces and verify that
re-adding D4 reconstructs the total labels at printed precision.

The implementation precedent is
[the C6 converter](../C6_wb97m_nodisp/prepare_data.py) and its
[release documentation](../C6_wb97m_nodisp/README.md).
C6's 266 accepted historical frames are reference evidence, not the new
2,000-cluster P10 dataset and not proof of equilibrium sampling.

## Required provenance for the new package

Retain raw .inp, .out, .property.txt and other artifacts, Slurm IDs/stdout,
ORCA version/module and converter source hashes. Record source geometry,
trajectory/frame/centre identity, reviewed analysis-window hash, acquisition
or held-out evaluation role, input/output hashes, explicit charge and spin,
source_method=wB97M-D4rev def2-TZVPD EnGrad RIJCOSX,
dft_functional_basis=wB97M/def2-TZVPD,
dispersion_removed=ORCA_printed_D4rev, absolute energy reference, and units.
Do not rewrite historical release manifests to describe this future batch.

## Submission status and migration

The active pointer selects `p10_foundation_1ns_10rep_2000dft_20261104T013941964400Z`.
Its configuration explicitly selects `orca_d4rev_subtraction`. Input generation
emits the named D4rev route; collection reuses the audited C6 converter after
checking exact echoed input, geometry, charge and singlet multiplicity. It
rejects missing corrections and unexpected nonlocal dispersion and verifies
energy/force reconstruction. Plans hash the shared converter source; package
and site manifests identify calculation and label conventions.

The old direct-route plan is preserved, not edited or bypassed. Its optional
pilot tooling remains historical; the selected subtraction route does not
require a fabricated direct-method verification record.

The corrected submission helper was invoked with Viper Python 3.10 from
`/viper/ptmp1/kshao/envs/mlip_clean/bin/python`; preflight passed the selected
route gate and stopped at missing `dft_package.json`, before sbatch.

A refreshed `origin/main` tree (zero divergence from HEAD) contains 1,000 old
P10 input geometries under
`codes/X_semiactivelearning/dft_launches/p10_2gpu_6h/P10p0000GPa/acquisition/inputs/`
and the historical 266 accepted labels. A filename inventory including ignored
files under `codes/` and `outputsfull/` found no new replica production
trajectories, selection.csv, cluster_quality.json or dft_package.json.
Geometries exist historically; the new reviewed 1,950+50 batch is missing.
Do not substitute or duplicate old transient geometries to reach this budget.

The planned Slurm topology is 200 array tasks, each running ten serial ORCA
calculations (2,000 calculations total), with at most 20 tasks concurrent.
No production job was submitted during this documentation update.
