# B1 Condense XYZ Transfer

This package streams the DAIS condition-production `.xyz` trajectories and writes
every 100th frame to a smaller transfer directory. It does not modify source
files and does not preprocess atom records; selected frames are copied
byte-for-byte from the input files.

Default DAIS paths:

- Input: `/dais/fs/scratch/kshao/mlip/outputsfull/conditionsproduction`
- Output: `/dais/fs/scratch/kshao/mlip/outputsfull/B1_conditionsproduction_stride100_xyz`
- Manifest: `B1_condense_manifest.csv` in the output root

Submit on DAIS:

```bash
cd /dais/fs/scratch/kshao/mlip/codes/B_conditionsproduction/B1_condense_xyz_transfer
sbatch B1_condense_conditionsproduction.slurm
```

The Slurm script requests the same GPU partition/GRES style as the production
base file because CPU-only jobs cannot be submitted. The work is mostly disk I/O,
so parallelism is across trajectories rather than within a single trajectory.

Useful overrides:

```bash
STRIDE=100 WORKERS=6 sbatch B1_condense_conditionsproduction.slurm
INPUT_ROOT=/path/to/conditionsproduction OUTPUT_ROOT=/path/to/condensed sbatch B1_condense_conditionsproduction.slurm
```
