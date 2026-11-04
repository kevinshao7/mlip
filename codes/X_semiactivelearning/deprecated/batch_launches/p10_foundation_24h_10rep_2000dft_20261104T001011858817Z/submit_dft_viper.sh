#!/bin/bash
set -euo pipefail
export MLIP_ROOT=${MLIP_ROOT:-/ptmp/kshao/mlip}
cd "$MLIP_ROOT"
"${PYTHON_COMMAND:-python3}" codes/X_semiactivelearning/p10_batch.py --plan "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_24h_10rep_2000dft_20261104T001011858817Z/plan.json" check-dft
mkdir -p "$MLIP_ROOT/outputsfull/X_semiactivelearning/p10_foundation_24h_10rep_2000dft_20261104T001011858817Z/slurm"
sbatch "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_24h_10rep_2000dft_20261104T001011858817Z/dft_2000_viper.slurm"
