#!/bin/bash
set -euo pipefail
export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
source "${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}/bin/activate"
cd "$MLIP_ROOT"
python codes/X_semiactivelearning/p10_batch.py --plan "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_1ns_10rep_2000dft_20261104T002630841603Z/plan.json" check-md
mkdir -p "$MLIP_ROOT/outputsfull/X_semiactivelearning/p10_foundation_1ns_10rep_2000dft_20261104T002630841603Z/slurm"
dependency=()
for segment in $(seq 1 "${MD_SEGMENTS:-6}"); do
    job=$(sbatch --parsable "${dependency[@]}" "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_1ns_10rep_2000dft_20261104T002630841603Z/md_10_replicas_dais.slurm")
    job=${job%%;*}
    echo "Prepared segment $segment submitted as array $job"
    dependency=(--dependency=afterok:$job)
done
