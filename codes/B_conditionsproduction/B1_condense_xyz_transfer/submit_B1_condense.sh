#!/bin/bash -l
set -euo pipefail

cd /dais/fs/scratch/kshao/mlip/codes/B_conditionsproduction/B1_condense_xyz_transfer
mkdir -p /dais/fs/scratch/kshao/mlip/outputsfull/slurm
sbatch B1_condense_conditionsproduction.slurm
