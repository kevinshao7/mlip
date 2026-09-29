# PolarMACE Naive Fine Tuning

This directory is a target-only, **naive** fine-tuning workflow for the
`polar-1-s` PolarMACE foundation model. It intentionally has no replay data
and does not enable MACE multihead fine tuning. The foundation weights are
used to initialize one `Default` head, which is then optimized against the
ORCA DFT target data only.

The authoritative data source is `data/target_all.xyz`. It includes the newly
obtained nitrogen-containing DFT structures. The current deterministic split
derived from that file is `181` training, `23` validation, and `23` test
structures. It is seeded and stratified by source family and nitrogen presence,
so validation and test contain representative nitrogen-bearing C3 frames; use
`target_train.xyz` and `target_valid.xyz` for training rather than an older
oxygen-only or delta-label split. The superseded 205/11/11 formula-level split
is preserved under `data/archive/` for prior-run reproducibility.

`../WORKFLOW_MAP.md` distinguishes this current target-only workflow from the
older oxygen-only multihead and delta experiments. Before preparing a new
experiment, run the explicit read-only audit:

```bash
python analyze_dataset_distribution.py
```

It writes `data/dataset_distribution/dataset_distribution_summary.json` and
`dataset_distribution_by_split.csv`, including per-split elemental/frame
balance, charge and spin distributions, formal-charge validation, duplicate
geometry checks, and cross-split leakage checks.

## Data preparation

Run on the Python installation that contains ASE and Torch:

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\prepare_data.py --workers 8
```

This optional preparation step reads completed ORCA outputs from
`outputsfull\\C_DFTproduction\\C_DFTproduction\\dft_outputs`, checking both
`FINAL SINGLE POINT ENERGY` and `ORCA TERMINATED NORMALLY`.  It writes
`data\\target_all.xyz` plus deterministic, seeded 80/10/10
train/validation/test splits stratified by source family and nitrogen presence.
This avoids a validation or test set that
contains only one chemistry family.

Energies and forces parsed by ASE are in eV and eV/Angstrom.  Each molecular
configuration is non-periodic (`pbc=False`), with `charge`, spin multiplicity
(`spin`), and a zero external field retained in the extended-XYZ metadata.
Before splitting, records whose stored `charge` disagrees with the project
formal-charge convention (`H=+1`, `N=-3`, `O=-2`) are rejected. The original
source is never overwritten: use `--source-target-all` with `--skip-orca` to
filter an existing extxyz dataset into this directory.
Training uses `--E0s=foundation`: the atomic-energy table embedded in the
`polar-1-s` MACE-POLAR checkpoint. This is the active and required default for
the current D2 workflow, which uses the same DFT standard as that foundation
model. Data preparation does not generate or select a custom E0 table unless
`--write-dft-e0s` is explicitly requested for a separate experiment. Do not
use `estimated`, `dft`, or an E0 JSON path in a standard D2 run.

By default, failed or incomplete ORCA outputs are reported and omitted.  Pass
`--strict` to reject a partial dataset.

The default training command uses this directory's stratified target splits.
To reproduce a split, rerun `prepare_data.py` with the same `--seed` (default
`3`). You can also provide explicit files with `--train-file` and
`--valid-file` after `--` to `launch_single_gpu.py`.

## Training

Run production training on one GPU (the current default physical GPU ID is `1`):

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\launch_single_gpu.py --gpu 1
```

The launcher accepts exactly one physical GPU ID, sets `CUDA_VISIBLE_DEVICES`,
and invokes `trainmace.py` directly. Distributed training is not supported in
this workflow. Training and validation batch sizes are both `1`.

The restored validated baseline uses `energy_weight=0.001`,
`forces_weight=100`, Adam `lr=0.0001`, EMA with decay `0.99999`, and MACE
Stage Two/SWA from epoch 15 with energy and force weights of 1 and 100,000,
respectively, at `swa_lr=0.001`.

The currently prepared next experiment changes only the Stage-Two learning
rate to the intermediate `swa_lr=0.0003` and runs for 100 epochs. It is the
active default, so start it with no extra training arguments:

```bash
python launch_single_gpu.py
```

## Experiment procedure — one change at a time

Every real `trainmace.py` invocation creates a fresh datetime-named directory
under `runs/`; no previous result is overwritten. Before MACE starts, it
appends the resolved wrapper settings, exact MACE command, run path, training
log path, checkpoint-model path, and compiled-model path to the one continuous
`hyperparameters.txt` file in this directory. It also writes
`runs/<run>/run_metadata.json` with the input-file SHA-256 hashes, Git revision,
Python executable, GPU visibility, cache path, resolved arguments, and MACE
command. A terminal `RUN END` record is automatically appended for both
successful and failed Python-level training attempts.

Follow this procedure for every iteration:

1. Begin from the validated baseline.
2. Change **one and only one** scientific or optimization factor.
3. State that factor with `--changed-parameter` and the expected effect with
   `--change-note`.
4. Compare its validation result only with the immediate parent baseline before
   proposing another modification.

For example, a learning-rate-only trial is:

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\launch_single_gpu.py --gpu 0 -- --lr 0.00005 --changed-parameter learning_rate --change-note "Halve Adam LR to test update stability."
```

Never combine changes to data, split, model architecture, foundation model,
optimizer, learning rate, loss weights, precision, or scheduler in one run.
Use `--dry-run` to inspect the command without creating a run directory.

Inspect the resolved MACE command without training:

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\launch_single_gpu.py --dry-run -- --max-num-epochs 1
```

Outputs are isolated in unique datetime-named directories under `runs\\`.
Downloads cache under `outputsfull\\.cache` through `XDG_CACHE_HOME`, rather
than the user home directory. Default settings preserve the target loss,
foundation model, precision, and batch sizes, while setting
`--multiheads_finetuning=False` and omitting all replay options.

## Evaluation

```powershell
C:\\Users\\shaoq\\AppData\\Local\\Programs\\Python\\Python312\\python.exe .\\mlip\\codes\\D2_naive\\evaluate.py --model .\\mlip\\codes\\D2_naive\\runs\\polar1s_naive_orca_dft_e0\\models\\polar1s_naive_orca_dft_e0.model
```
