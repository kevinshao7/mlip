#!/bin/bash -l
#SBATCH --job-name=temperature_ramp_base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.err

set -e

module purge
module load anaconda/3/2023.03

source /ptmp/kshao/envs/mlip_clean/bin/activate
export PYTHONNOUSERSITE=1

export MLIP_MACE_DEVICE=${MLIP_MACE_DEVICE:-cuda}

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MLIP_MACE_DEVICE=${MLIP_MACE_DEVICE}"
echo "CHECKPOINT_XYZ=${CHECKPOINT_XYZ:-default r09_hot_w pressure trajectory}"
echo "CHECKPOINT_FRAME=${CHECKPOINT_FRAME:--1}"

cd /ptmp/kshao/mlip/codes/7_24_temperatureramp
# Local GPU run without Slurm: source ~/env/bin/activate && cd ~/mlip/codes/7_24_temperatureramp && CHECKPOINT_XYZ=/path/to/checkpoint.xyz CUDA_VISIBLE_DEVICES=0 MLIP_MACE_DEVICE=cuda python TemperatureRampbase.py
srun python TemperatureRampbase.py
