#!/bin/bash -l
#SBATCH --job-name=D2_finetune
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000
#SBATCH --time=24:00:00
#SBATCH --chdir=/dais/fs/scratch/kshao/mlip/codes/D2_naive
#SBATCH --output=/dais/fs/scratch/kshao/mlip/outputsfull/slurm/%x_%j.out
#SBATCH --error=/dais/fs/scratch/kshao/mlip/outputsfull/slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ks2120@cam.ac.uk

set -euo pipefail

# Reuse the working DAIS GPU environment from B_conditionsproduction.
module purge
unset MPI_ROOT CUDA_ROOT LD_LIBRARY_PATH
module load gcc/14
module load cuda/12.8

export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
export MLIP_ENV=${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$MLIP_ROOT/outputsfull/.cache"
mkdir -p "$XDG_CACHE_HOME"
source "$MLIP_ENV/bin/activate"
cd "$MLIP_ROOT/codes/D2_naive"

module list 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
srun --ntasks=1 --kill-on-bad-exit=1 python -c 'import ase, mace, torch, graph_longrange; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count() == 1, "Expected one allocated GPU"; print("DAIS training environment OK")'
# The Python launcher retains Slurm visibility and records each real run.
srun --ntasks=1 --kill-on-bad-exit=1 python launch_single_gpu.py "$@"
