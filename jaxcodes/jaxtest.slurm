#!/bin/bash -l
#SBATCH --job-name=jax

#SBATCH --nodes=1

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=125G
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2023.03
module load git/2.50
module load cuda/12.8

export CUDA_LAUNCH_BLOCKING=1

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
cd /ptmp/kshao/jax/jaxmd-cli
source .venv/bin/activate
jaxmd-run /ptmp/kshao/jax/jaxmd-cli/codes/maceoff_water_npt.toml
