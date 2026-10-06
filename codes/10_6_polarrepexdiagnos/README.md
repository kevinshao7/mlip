# MACE-POLAR / MACE-OFF replica-exchange diagnostic

This workflow samples 100 configurations from all 16 replica-exchange
trajectories, balanced across lambda and simulation time, then compares
MACE-POLAR `polar-1-m` against MACE-OFF `medium`.

The default sampler omits the leading 20% of each trajectory as equilibration.
It assigns 6 or 7 configurations to each replica and selects the midpoint of
equal-width time strata. The resulting observations are still correlated MD
samples; parity metrics are exploratory model-disagreement diagnostics, not
independent-error confidence estimates.

Run from the repository root:

```powershell
& C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\10_6_polarrepexdiagnos\sample_repex_frames.py
& C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\10_6_polarrepexdiagnos\compare_mace_polar_off.py --device cuda
```

Generated data go to `mlip/outputsfull/10_6_polarrepexdiagnos`. The comparison
writes the sampled structures and manifest, both model prediction trajectories,
centered-energy and force-component parity PNGs, an energy CSV, and JSON metrics.

The energy plot subtracts each model's mean energy over the fixed-composition
sample. This is necessary because MACE-POLAR and MACE-OFF can have different
atomic reference-energy zeros. Raw total energies and the two removed means are
retained in the CSV/JSON. Forces require no reference alignment.

The system convention is periodic boundaries, total charge 0, spin multiplicity
1 (singlet), energies in eV, positions in Å, and forces in eV/Å. Override model,
sampling, charge, spin, device, or precision choices through `--help`; these are
never inferred from trajectory coordinates.
