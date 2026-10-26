# Fine-tuning megadataset

`finetuning_megadataset.jsonl` is the repository-tracked, append-only
registry for every MACE fine-tuning attempt. One JSON object per line records
the immutable run directory, method, foundation model, full hyperparameters,
MACE command, Git revision, train/validation/replay paths and SHA-256 hashes,
structure/atom counts, campaign lineage when present, initial/final validation
loss and energy/force RMSE, start/end time, duration, and terminal status.

The following launchers write a terminal record automatically:

- `D2_naive/trainmace.py` — target-only and semi-active-learning batches
- `D_MHFT/trainmace.py` — multihead replay fine tuning
- `7_29_finetunepolar/runtrain.py` — legacy multihead replay launcher

Backfill or validate the Git-tracked registry with one command:

```bash
/home/kevinsh/env/bin/python /home/kevinsh/mlip/codes/backfill_finetuning_megadataset.py
```

Use `--reset` only to rebuild the generated registry from immutable historical
`run_metadata.json` files. New terminal run records are append-only and are
deduplicated by immutable run directory. Commit the changed JSONL together
with the corresponding code change after reviewing it.
