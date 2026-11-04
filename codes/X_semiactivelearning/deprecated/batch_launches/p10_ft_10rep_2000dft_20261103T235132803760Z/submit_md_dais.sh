#!/bin/bash
set -euo pipefail
export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
cd "$MLIP_ROOT"
source "${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}/bin/activate"
python codes/X_semiactivelearning/p10_batch.py --plan "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_ft_10rep_2000dft_20261103T235132803760Z/plan.json" check-md
mkdir -p "$MLIP_ROOT/outputsfull/X_semiactivelearning/p10_ft_10rep_2000dft_20261103T235132803760Z/slurm"
sbatch "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_ft_10rep_2000dft_20261103T235132803760Z/md_10_replicas_dais.slurm"
