# Reduce fine-condition trajectories

`reduce_xyz_stride.py` streams each `production*.xyz` trajectory below
`outputsfull/conditionsfine`, retains frames 0, 100, 200, ..., and writes the
result under the matching run directory in `outputsfull/conditionsfinereduced`.
Checkpoint XYZ files are deliberately excluded.

The B2 production settings are a 0.5 fs MD step and one saved frame every five
steps. Therefore input frames are 2.5 fs apart, and the default reduced frames
are 250 fs (0.25 ps) apart. Every output XYZ comment records
`source_frame_index`, `source_time_fs`, `source_frame_interval_fs`,
`reduction_stride`, and `reduced_frame_interval_fs`. All original XYZ fields
and atom lines are preserved.

On the system containing the production outputs:

```bash
cd /dais/fs/scratch/kshao/mlip
python codes/B2_conditionsfine/reduce_xyz_stride.py
```

Set `MLIP_ROOT` if the checkout is elsewhere, or pass `--input-root` and
`--output-root`. Existing reduced trajectories are protected unless `--force`
is supplied. A summary is written to
`outputsfull/conditionsfinereduced/reduction_manifest.csv`.
