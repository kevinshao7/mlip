# Replica-exchange thermo plots

`process_repex.py` reads `thermo.csv` from every `replica_*` directory and creates
one figure per replica. Each figure contains time series and normalized
autocorrelation plots for density, temperature, and total energy. Time and lag
time are shown in picoseconds.

Run with the configured default paths:

```powershell
python .\mlip\codes\7_19_processrepex\process_repex.py
```

Optionally specify another run directory, output directory, or maximum lag
(expressed as a number of saved samples):

```powershell
python .\mlip\codes\7_19_processrepex\process_repex.py <run-directory> --output-dir <plot-directory> --max-lag 500
```

By default, autocorrelations are displayed through half the trajectory length,
where they are less dominated by the small number of pairs at the longest lags.

## Atom composition plots

`plot_atom_compositions.py` uses the local `mlip/src/asemolec` checkout and the
Python executable from the replica-processing command:

```powershell
& C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\7_19_processrepex\plot_atom_compositions.py
```

By default it analyzes each replica's `minimized.pdb` and saves a pie chart in
`plots`.
To sample full trajectories from `trajectory.nc`, use:

```powershell
& C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\7_19_processrepex\plot_atom_compositions.py --source trajectory --stride 100
```

## Methanol-water RDFs

`plot_methanol_water_rdf.py` computes RDFs only for the fully interacting
replica, `replica_15_lambda_1.0000_el_1.0000`. It plots methanol carbon against
all oxygens, including the methanol oxygen and all water oxygens:

```powershell
& C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\7_19_processrepex\plot_methanol_water_rdf.py
```

The RDF outputs are saved as `carbon_oxygen_rdf_replica_15.csv` and
`carbon_oxygen_rdf_replica_15.png` in `plots`.
