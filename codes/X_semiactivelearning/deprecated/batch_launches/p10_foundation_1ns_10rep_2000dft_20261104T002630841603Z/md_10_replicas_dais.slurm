#!/bin/bash -l
#SBATCH --job-name=X_P10_foundation_replica
#SBATCH --partition=gpu1
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dais/fs/scratch/kshao/mlip/outputsfull/X_semiactivelearning/p10_foundation_1ns_10rep_2000dft_20261104T002630841603Z/slurm/md_%A_%a.out
#SBATCH --error=/dais/fs/scratch/kshao/mlip/outputsfull/X_semiactivelearning/p10_foundation_1ns_10rep_2000dft_20261104T002630841603Z/slurm/md_%A_%a.err
set -euo pipefail
module purge
unset MPI_ROOT CUDA_ROOT LD_LIBRARY_PATH
module load gcc/14
module load cuda/12.8
export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
export MLIP_ENV=${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$MLIP_ROOT/outputsfull/.cache"
export PATH="$MLIP_ROOT/packmol:$PATH"
source "$MLIP_ENV/bin/activate"
cd "$MLIP_ROOT"

srun --ntasks=1 --kill-on-bad-exit=1 python codes/X_semiactivelearning/p10_batch.py --plan "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_1ns_10rep_2000dft_20261104T002630841603Z/plan.json" run-md --index "${SLURM_ARRAY_TASK_ID}"
