# C2 Atomization DFT

This directory regenerates isolated-atom ORCA reference energies for:

- `H`
- `O`
- `N`
- `S`

It uses FairChem's ORCA writer and stages FairChem's `def2-tzvpd.bas`.

That keeps the DFT settings aligned with the production cluster workflow without
hand-written ORCA input templates. Generated inputs use `%pal nprocs 8 end`.

## Important

Set `ORCA_COMMAND` to the ORCA executable or wrapper available on the machine.
`run_all.sh` defaults to `orca_qc`.

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

On Windows PowerShell, use the native runner:

```powershell
Set-Location C:\Users\shaoq\Documents\Mainz\mlip\codes\C2_atomizationDFT
.\run_all.ps1
```

Rewrite the shared reference CSV after the runs finish:

```bash
cd /home/kevinsh/mlip/codes/C2_atomizationDFT
python extract_atomization_energies.py
```

The extractor overwrites:

- `/home/kevinsh/mlip/codes/7_7b_clustervalidation/atomizationenergies.txt`
