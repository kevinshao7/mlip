# C5 D4rev atomic reference energies

This workflow computes isolated neutral-atom ORCA reference energies for H, O,
N, and S using the same production method currently used by the semi-active
learning DFT workflow:

```text
wB97M-D4rev/def2-TZVPD, EnGrad, RIJCOSX, ORCA 6, 20 MPI ranks
```

These calculations are non-periodic, zero-temperature electronic-structure
calculations; no thermodynamic ensemble or real-space cutoff applies. Each
input explicitly records charge and spin multiplicity. The reference species
are neutral isolated atoms, so their formal charge is `0`; their ground-state
multiplicities are H=2, N=4, O=3, and S=3. This neutral-atom convention is
specific to atomic energy references and must not be substituted for the
project's composition-based formal-charge rule for molecular clusters.

The route deliberately uses ORCA 6's built-in `def2-TZVPD` keyword, matching
`X_semiactivelearning`; it does not stage the older FairChem basis file used by
`C2_atomizationDFT`.

## Prepare and validate

From this directory:

```text
python generate_atom_inputs.py
python generate_atom_inputs.py --check
```

Generation is idempotent when files match. It refuses to replace a differing
input or manifest unless `--force` is explicitly supplied.

## Run ORCA sequentially

The Python runner defaults to `orca_qc`. It starts H, O, N, and S in that
order, waits for each process to finish and validates its output before
starting the next calculation. The jobs therefore run sequentially, while
each individual ORCA job uses 20 MPI ranks.

```text
python run_all.py
```

To continue after an interrupted sequence, retain normally completed outputs
and use:

```text
python run_all.py --resume
```

`--resume` skips only outputs containing both `FINAL SINGLE POINT ENERGY` and
`ORCA TERMINATED NORMALLY`. An incomplete output is never silently rerun or
overwritten; archive it after diagnosis, then invoke the runner again. To use
an executable other than the default, pass `--orca-command NAME`.

The Python process pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OPENBLAS_NUM_THREADS` to one. Parallelism comes from the `%pal nprocs 20 end`
block in every ORCA input, preventing nested thread oversubscription.

The PowerShell runner is also available for Windows and reads the same inputs:

```powershell
.\run_all.ps1
```

## Extract energies

After all four calculations finish:

```text
python extract_atomization_energies.py
```

Hartree energies are converted using `1 Eh = 27.211386245988 eV` and written
to `atomizationenergies_D4rev.txt`. The same extraction writes
`e0s_wb97m_d4rev.json` in MACE's atomic-number format; the active fine-tuning
wrapper requires both files to agree exactly. The extractor also requires
normal ORCA termination and refuses to replace either result without `--force`.
It intentionally does not overwrite the older shared `wB97M-V` reference file.
