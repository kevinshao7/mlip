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
