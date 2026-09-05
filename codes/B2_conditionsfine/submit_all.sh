#!/bin/bash -l
set -euo pipefail

cd /dais/fs/scratch/kshao/mlip/codes/B2_conditionsfine
mkdir -p /dais/fs/scratch/kshao/mlip/outputsfull/slurm

if ! compgen -G "expand/production_*.sh" > /dev/null; then
    python make_production_runs.py
fi

for script in expand/production_*.sh; do
    sbatch "$script"
done
