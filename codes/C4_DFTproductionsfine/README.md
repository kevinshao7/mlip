# C4 fine-condition DFT pilot

This workflow prepares four ORCA single-point/gradient inputs from the reduced
fine-condition trajectories at 17.7828 GPa. It does not execute ORCA or submit
jobs.

The default pilot uses only `P17p7828GPa_R0p5` (NH3/H2O ratio 0.5), with one
frame at each of 20%, 40%, 60%, and 80% through the reduced trajectory. A
deterministic target molecule is selected in each frame. Molecules are defined
by the exact graph construction used by
`B_conditionsproduction/plot_species_time_series.py`: heavy-atom covalent
bonds plus unconditional assignment of each H to its nearest H, O, or N atom.
Each DFT cluster contains that target component and the five other molecular
components with the smallest minimum-image interatomic distance to it.

Formal charge uses the project H/N/O rule (`H=+1`, `N=-3`, `O=-2`) and is
recomputed when the ORCA inputs are prepared. Multiplicity is 1 for an even
electron count and 2 for an odd count. Inputs use wB97M-V/def2-TZVPD with
EnGrad, matching C3.

From the repository root:

```powershell
python .\mlip\codes\C4_DFTproductionsfine\extract_molecule_clusters.py
python .\mlip\codes\C4_DFTproductionsfine\prepare_orca_inputs.py
```

The second command only writes four `.inp` files and copies `def2-tzvpd.bas`
into `expand`. It never calls ORCA. Existing `.inp` files are protected unless
`--force` is passed.
