# MACE-POLAR / MACE-OFF replica-exchange diagnostic

> [!IMPORTANT]
> `polar-1-s` is retired for new work. Use `polar-1-l`; any older references
> are historical and remain only for reproducibility.

This workflow samples 300 configurations from all 16 replica-exchange
trajectories, balanced across lambda and simulation time, then compares
MACE-POLAR `polar-1-m` against MACE-OFF `medium`.

The default sampler omits the leading 20% of each trajectory as equilibration.
It assigns 18 or 19 configurations to each replica and selects the midpoint of
equal-width time strata. The resulting observations are still correlated MD
samples; parity metrics are exploratory model-disagreement diagnostics, not
independent-error confidence estimates.

Start with the 10-frame diagnostic from the repository root. It selects the
midpoint of ten equal-width strata from `representative_300.xyz`, so it spans
the stored lambda/time sample rather than evaluating its first ten frames:

```powershell
& C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\10_6_polarrepexdiagnos\compare_mace_polar_off.py --device cuda
```

The sampler writes the GitHub-tracked `representative_300.xyz` and its CSV
manifest beside these scripts. Model predictions and plots remain generated
data under `mlip/outputsfull/10_6_polarrepexdiagnos/parity/test_10_frames`.
That directory contains a settings fingerprint, so predictions from another
input, frame count, model, charge/spin, device, or precision cannot be reused
silently. Use a separate output directory for a 300-frame production diagnostic:

```powershell
& C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\10_6_polarrepexdiagnos\compare_mace_polar_off.py --device cuda --n-frames 300 --output-dir .\mlip\outputsfull\10_6_polarrepexdiagnos\parity\full_300
```

The energy plot subtracts each model's mean energy over the fixed-composition
sample. This is necessary because MACE-POLAR and MACE-OFF can have different
atomic reference-energy zeros. Raw total energies and the two removed means are
retained in the CSV/JSON. Forces require no reference alignment.

The stored composition is one neutral methanol plus 264 water molecules
(`CH532O265`), so the declared total charge is 0 and spin multiplicity is 1.
All frames must have fully periodic boundary conditions. The comparison rejects
frames with a minimum-image pair distance below 0.600 Å before evaluating
either model; these overlaps are outside the intended physical ensemble and
can dominate a parity statistic. Energies are eV, positions Å, and forces
eV/Å. MACE-POLAR uses native float32 and MACE-OFF native float64 by default;
both settings are recorded. Override model, sampling, charge, spin, device, or
precision choices through `--help`; they are never inferred from coordinates.
