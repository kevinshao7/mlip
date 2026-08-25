# PolarMACE Multihead Fine Tuning

This folder prepares ORCA DFT outputs and runs multihead fine tuning of
`polar-1-s` from the local `mlip/mace` checkout.

## Data Prep

Run with the Python 3.12 interpreter that already has ASE/Torch installed:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\D_MHFT\prepare_data.py
```

The default ORCA extraction uses `--workers 8`. To make that explicit:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\D_MHFT\prepare_data.py --workers 8
```

This writes:

- `data\target_all.xyz` from `outputsfull\C_DFTproduction\C_DFTproduction\dft_outputs`
- `data\target_train.xyz`, `data\target_valid.xyz`, `data\target_test.xyz`
- `data\target_e0s.json` from `codes\7_7b_clustervalidation\atomizationenergies.txt`
- `data\omol_replay_unlabeled.xyz` from either an OMol pickle archive or the
  official sharded ASE-LMDB (`.aselmdb`) archives

The ORCA converter uses ASE's ORCA parser, so energies are written in eV and
forces in eV/Angstrom. The target labels are `REF_energy` and `REF_forces`.
PolarMACE metadata fields `charge`, `spin`, and `external_field` are also
written.

The default behavior writes a partial dataset if some ORCA outputs are missing,
incomplete, or unparsable. Problem files are printed as non-fatal warnings and
are not included in the output. Use `--strict` when you want conversion to exit
nonzero instead of writing a partial dataset.

## Replay Data

The OMOL archive currently present in `outputsfull\C1_omol` contains geometry
input pickles, not true labeled replay energies/forces. The default training
command therefore uses `--pseudolabel_replay=True`, so MACE labels replay
geometries with the starting `polar-1-s` model. `trainmace.py` now detects
whether `REF_energy` and `REF_forces` are present: labeled original training
data is used directly, while unlabeled data is pseudolabeled.

Official OMol25 downloads contain sharded ASE-LMDB files. They can be passed
without manually extracting the full archive; preparation extracts only enough
shards to satisfy `--max-replay-configs`:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\D_MHFT\prepare_data.py --skip-orca --omol-archive 'C:\path\to\train_4M.tar.gz' --max-replay-configs 10000
```

Do not use the OMol25 `test` archive as labeled replay data. Its rows contain
structures and metadata but no DFT energy/force labels; auto mode will therefore
pseudolabel it. Use a labeled training archive (for example Train 4M) to replay
the original OMol25 training labels.

If you download a true labeled replay `.xyz` or `.extxyz` from the
MACE-foundations releases, pass it directly:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\D_MHFT\trainmace.py --pt-train-file C:\path\to\labeled_replay.extxyz --pseudolabel-replay False
```

## Training

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\D_MHFT\trainmace.py
```

Useful dry run:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\D_MHFT\trainmace.py --dry-run
```

Outputs go under `runs\polar1s_mhft_orca`.

## Evaluation

After training, evaluate the target head:

```powershell
C:\Users\shaoq\AppData\Local\Programs\Python\Python312\python.exe .\mlip\codes\D_MHFT\evaluate.py --model .\mlip\codes\D_MHFT\runs\polar1s_mhft_orca\models\polar1s_mhft_orca.model
```

## Notes

- `polar-1-s` is the default foundation model.
- Target elements are configured as `[1, 7, 8, 16]` for H/N/O/S. Change
  `--atomic-numbers` if target data later includes other elements.
- MACE's multihead guide recommends explicit E0s for fine tuning; this workflow
  uses your omegaB97 atomization energies rather than `average`.
