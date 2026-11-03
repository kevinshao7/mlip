#!/bin/bash -l
#SBATCH --job-name=X_md
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000
#SBATCH --time=24:00:00
#SBATCH --chdir=/dais/fs/scratch/kshao/mlip/codes/X_semiactivelearning
#SBATCH --output=/dais/fs/scratch/kshao/mlip/outputsfull/slurm/%x_%j.out
#SBATCH --error=/dais/fs/scratch/kshao/mlip/outputsfull/slurm/%x_%j.err
set -euo pipefail
module purge
unset MPI_ROOT CUDA_ROOT LD_LIBRARY_PATH
module load gcc/14
module load cuda/12.8
export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
export MLIP_ENV=${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$MLIP_ROOT/outputsfull/.cache"
export PATH="$MLIP_ROOT/packmol:$PATH"
mkdir -p "$XDG_CACHE_HOME"
source "$MLIP_ENV/bin/activate"
cd "$MLIP_ROOT/codes/X_semiactivelearning"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count() == 1, "Expected one assigned GPU"'
pressure=${1:-10}
srun --ntasks=1 --kill-on-bad-exit=1 python campaign.py md --pressure "$pressure" --run
