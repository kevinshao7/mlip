#!/bin/bash -l
set -euo pipefail

cd /dais/fs/scratch/kshao/mlip/codes/B_conditionsproduction
mkdir -p /dais/fs/scratch/kshao/mlip/outputsfull/slurm

if ! compgen -G "expand/production_*.sh" > /dev/null; then
    echo "No generated Slurm scripts found under expand/."
    echo "Regenerating with make_production_runs.py..."
    python make_production_runs.py
fi

if ! compgen -G "expand/production_*.sh" > /dev/null; then
    echo "ERROR: still no expand/production_*.sh files found after regeneration." >&2
    echo "Check that this directory contains NPTMACEproduction_base.py, NPTMACEproduction_base.slurm, and make_production_runs.py." >&2
    exit 1
fi

for script in expand/production_*.sh; do
    sbatch "$script"
done
