# 8_16 Process DFT

Process the ORCA outputs from
`outputsfull/C_DFTproduction/C_DFTproduction/dft_outputs` into frame-level and
atom-level reference files for later MLIP parity plots.

Run from the repository root:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\A_parityplot\8_16_processDFT\process_dft_outputs.py
```

Outputs go to:

```text
mlip/outputsfull/C_DFTproduction/C_DFTproduction/processed_dft_outputs/
```

Generated files:

- `C_DFTproduction_singlepoints.csv`: one row per frame with status, energy, and failure flags.
- `C_DFTproduction_forces.csv`: one row per atom for complete frames, including positions, gradients, and forces.
- `C_DFTproduction_complete.extxyz`: successful frames only, with `energy_eV` and per-atom `forces`.
- `C_DFTproduction_stats.json`: batch-level counts and output paths.

The script reads geometry and charge/multiplicity from the generated ORCA input
files in `mlip/codes/C_DFTproduction/expand/` and joins them with the
ORCA `.out` files. Incomplete or crashed jobs are kept in the summary CSV with
`status=incomplete` and a JSON `issues` field instead of aborting the batch.
