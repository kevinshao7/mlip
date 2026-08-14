#!/bin/bash -l
set -euo pipefail

cd /dais/fs/scratch/kshao/mlip/codes/B_conditionsproduction
mkdir -p /dais/fs/scratch/kshao/mlip/outputsfull/slurm

for script in expand/production_*.sh; do
    sbatch "$script"
done
