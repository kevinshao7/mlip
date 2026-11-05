#!/bin/bash -l
#SBATCH --job-name=X_P10_wB97M_2000
#SBATCH --array=0-199%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/X_semiactivelearning/p10_foundation_1ns_10rep_2000dft_20261104T013941964400Z/slurm/dft_%A_%a.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/X_semiactivelearning/p10_foundation_1ns_10rep_2000dft_20261104T013941964400Z/slurm/dft_%A_%a.err
set -euo pipefail
module purge
module load orca/6.1.1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MLIP_ROOT=${MLIP_ROOT:-/ptmp/kshao/mlip}
export ORCA_COMMAND=${ORCA_COMMAND:-orca}
export PYTHON_COMMAND=${PYTHON_COMMAND:-python3}
cd "$MLIP_ROOT"
"$PYTHON_COMMAND" codes/X_semiactivelearning/p10_batch.py --plan "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_1ns_10rep_2000dft_20261104T013941964400Z/plan.json" run-dft-group --index "${SLURM_ARRAY_TASK_ID}"
