# DFT tasks: published 10 GPa clusters on Viper

The new CPU pipeline uses the published release selected by
`../partial_clusters_active.json`, without requiring completed 1 ns trajectories
or changing the original MD plan. `../cpu_dft_active.json` selects the prepared
CPU package. This is separate from the older P10 array launcher.

From `codes/X_semiactivelearning/`:

```bash
../../.venv/bin/python dft/viper_cpu.py check
bash submitcpu.sh --dry-run
# Submit only when you choose to launch:
bash submitcpu.sh
# After calculations finish, validate and collect electronic-only labels:
../../.venv/bin/python dft/viper_cpu.py collect
```

The prepared package contains **200 separate Slurm jobs**, each processing
**10 clusters sequentially**, with **20 CPUs per ORCA calculation** via
`%pal nprocs 20`. Defaults: Viper `general`, 64 GiB, 24 hours, ORCA module
`orca/6.1.1`. Viper has no partition named `cpu`; `sinfo` and `scontrol` on
2026-11-05 confirmed a 24-hour maximum for `general` and `small`. Ten clusters
finishing within one allocation has not been benchmarked. Each job reserves
one task with 20 CPUs; ORCA's driver launches its MPI workers, and is invoked
by absolute executable path without an extra `srun`/`mpirun` prefix. Threads
per worker are limited to one. No GPU allocation or MD/model is needed.

Method: `! wB97M-D4rev def2-TZVPD EnGrad RIJCOSX`, isolated nonperiodic
clusters, formal charge H-3N-2O recomputed for every cluster and matched to
XYZ/manifest/audit/input, explicit verified closed-shell singlet. The existing
X output validator and C6 converter remove printed D4 energy and gradient.
Labels are **absolute electronic energies in eV** and **forces in eV/Angstrom**;
no atomic-reference subtraction. Both normal termination and final energy,
charge/spin, echoed input, geometry, method and gradient checks must pass.

Groups 000-194 contain the 1950 acquisition clusters; 195-199 contain the
50 evaluation clusters. Source metadata and temporal partitions are retained
in collected extxyz files. Never train on evaluation data. Adjacent frames
and repeated centres from one frame are correlated.

Inputs, manifests and `.slurm` scripts live in dated `dft/batches/` folders.
All runtime ORCA stdout, raw inputs, property files, scratch, exit codes,
submission records and Slurm logs stay under
`outputsfull/X_semiactivelearning/P10GPa/cpu_dft_results/<batch>/`.
Existing calculation attempts are never overwritten or retried automatically.
The submission helper validates the full package before its first `sbatch`,
records every accepted job ID, and skips recorded submissions on another
invocation. Ambiguous or failed submission attempts stop for inspection.
`--dry-run` only prints the 200 commands and does not call `sbatch`.

Preparation does not submit jobs. To prepare a fresh package after code or
resource changes, run `../../.venv/bin/python dft/viper_cpu.py prepare`.
Overrides are available for `--partition`, `--walltime`, and `--memory-gib`;
active prepared resources are frozen in `package.json`. Source/input hashes
prevent silently using modified code or geometries; prepare a new batch after
changes. Old MD bundles and extraction snapshots remain untouched.
