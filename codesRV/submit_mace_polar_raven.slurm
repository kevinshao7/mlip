#!/bin/bash -l
#SBATCH -J mace_polar_ft
#SBATCH -o ./slurm-%x.%j.out
#SBATCH -e ./slurm-%x.%j.err
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --mail-type=none

set -euo pipefail

# Raven uses environment modules. Check exact versions with:
#   module avail cuda
#   module avail anaconda
module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2023.03
module load git/2.50
mmodule load cuda/13.2

# Avoid CPU oversubscription from PyTorch/NumPy/OpenMP.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# More robust CUDA/PyTorch behavior on clusters.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

# Activate your environment. Adjust path if your venv has a different name/location.
# Example setup before submitting:
#   python3 -m venv env
#   source env/bin/activate
#   pip install --upgrade pip setuptools wheel
#   pip install "mace-torch @ git+https://github.com/ACEsuit/mace.git@main"
#   pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git
source ./env/bin/activate

# Optional: keep caches and MACE downloads somewhere persistent, not on compute-node tmp.
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PWD/.cache}
export MACE_RUN_NAME=${MACE_RUN_NAME:-polar_ft_1m}
export MACE_TRAIN_FILE=${MACE_TRAIN_FILE:-./ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz}
export MACE_FOUNDATION_MODEL=${MACE_FOUNDATION_MODEL:-polar-1-s}
export MACE_MAX_EPOCHS=${MACE_MAX_EPOCHS:-20}
export MACE_BATCH_SIZE=${MACE_BATCH_SIZE:-1}
export MACE_DTYPE=${MACE_DTYPE:-float32}

mkdir -p logs models results "$XDG_CACHE_HOME"

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# srun is the Slurm launcher. For this single-process PyTorch job, one task owns one GPU.
srun python ./train_mace_hpc.py

echo "Job finished at $(date)"
