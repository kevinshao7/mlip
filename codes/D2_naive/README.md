# PolarMACE Naive Fine Tuning

This directory is a target-only, **naive** fine-tuning workflow for the
`polar-1-s` PolarMACE foundation model.  It intentionally has no replay data
and does not enable MACE multihead fine tuning.  The foundation weights are
used to initialize one `Default` head, which is then optimized against the
ORCA DFT target data only.

## Data preparation

Run on the Python installation that contains ASE and Torch:

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\prepare_data.py --workers 8
```

This optional preparation step reads completed ORCA outputs from
`outputsfull\\C_DFTproduction\\C_DFTproduction\\dft_outputs`, checking both
`FINAL SINGLE POINT ENERGY` and `ORCA TERMINATED NORMALLY`.  It writes
`data\\target_all.xyz`, plus deterministic contiguous train/validation/test
splits and `data\\target_dft_e0s.json`.

Energies and forces parsed by ASE are in eV and eV/Angstrom.  Each molecular
configuration is non-periodic (`pbc=False`), with `charge`, spin multiplicity
(`spin`), and a zero external field retained in the extended-XYZ metadata.
The data-preparation utility can write omegaB97 DFT isolated-atom energies for
reproducible energy comparisons. Training does not use that custom table by
default: it uses the `foundation` E0 mode, which retains the atomic-energy
table embedded in the MACE-POLAR checkpoint. `estimated` instead refits E0s
from foundation-model predictions on the target training data. Use `--e0s dft`
only when the isolated-atom ORCA reference is explicitly required.

By default, failed or incomplete ORCA outputs are reported and omitted.  Pass
`--strict` to reject a partial dataset.

The default training command instead reuses the completed target train and
validation split in `codes\\D_MHFT\\data`.  This avoids copying generated data
and makes the naive and multihead results directly comparable.  If you prepare
a separate split in this directory, pass it explicitly with `--train-file` and
`--valid-file` after `--` to `launch_single_gpu.py`.

## Training

Run production training on one GPU (the default physical GPU ID is `0`):

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\launch_single_gpu.py --gpu 0
```

The launcher accepts exactly one physical GPU ID, sets `CUDA_VISIBLE_DEVICES`,
and invokes `trainmace.py` directly. Distributed training is not supported in
this workflow. Training and validation batch sizes are both `1`.

The defaults use `energy_weight=0.001` and `forces_weight=100`, retaining a
100,000:1 force:energy ratio without globally rescaling the prior force-loss
scale. The learning rate remains 0.01. MACE SWA (Stage Two) is always enabled:
it starts at epoch 15 with energy and force weights of 1 and 100,000,
respectively, and uses MACE's default Stage Two learning rate of 0.001. Change
the start or weights with `--start-swa`, `--swa-energy-weight`, and
`--swa-forces-weight` after the launcher's `--` separator.

Inspect the resolved MACE command without training:

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\launch_single_gpu.py --dry-run -- --max-num-epochs 1
```

Outputs are isolated under `runs\\polar1s_naive_orca_dft_e0`.  Downloads cache
under `outputsfull\\.cache` through `XDG_CACHE_HOME`, rather than the user home
directory.  Default settings preserve the D_MHFT target loss, data split,
foundation model, precision, and batch sizes, while setting
`--multiheads_finetuning=False` and omitting all replay options.

## Evaluation

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\evaluate.py --model .\\mlip\\codes\\D2_naive\\runs\\polar1s_naive_orca_dft_e0\\models\\polar1s_naive_orca_dft_e0.model
```
