# C2 Atomization DFT

This directory regenerates isolated-atom ORCA reference energies for:

- `H`
- `O`
- `N`
- `S`

It uses the exact ORCA input template and basis file from:

- `/home/kevinsh/mlip/codes/C_DFTproduction/base.inp`
- `/home/kevinsh/mlip/codes/C_DFTproduction/def2-tzvpd.bas`

That keeps the DFT settings aligned with the production cluster workflow while
avoiding a separate hand-written atom-specific template.

## Important

On this HPC machine, the correct execution command is `orca_qc`, not `orca`.
`run_all.sh` defaults to `ORCA_COMMAND=orca_qc` for that reason.

## Usage

Generate the isolated-atom inputs and stage the basis file:

```bash
cd /home/kevinsh/mlip/codes/C2_atomizationDFT
python generate_atom_inputs.py
```

Run the four ORCA jobs locally with the HPC wrapper command:

```bash
cd /home/kevinsh/mlip/codes/C2_atomizationDFT
bash run_all.sh
```

Rewrite the shared reference CSV after the runs finish:

```bash
cd /home/kevinsh/mlip/codes/C2_atomizationDFT
python extract_atomization_energies.py
```

The extractor overwrites:

- `/home/kevinsh/mlip/codes/7_7b_clustervalidation/atomizationenergies.txt`
