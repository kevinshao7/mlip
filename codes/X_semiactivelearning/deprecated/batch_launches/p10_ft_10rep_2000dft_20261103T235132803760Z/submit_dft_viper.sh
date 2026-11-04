#!/bin/bash
set -euo pipefail
export MLIP_ROOT=${MLIP_ROOT:-/ptmp/kshao/mlip}
cd "$MLIP_ROOT"
"${PYTHON_COMMAND:-python3}" codes/X_semiactivelearning/p10_batch.py --plan "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_ft_10rep_2000dft_20261103T235132803760Z/plan.json" check-dft
mkdir -p "$MLIP_ROOT/outputsfull/X_semiactivelearning/p10_ft_10rep_2000dft_20261103T235132803760Z/slurm"
sbatch "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_ft_10rep_2000dft_20261103T235132803760Z/dft_2000_viper.slurm"
