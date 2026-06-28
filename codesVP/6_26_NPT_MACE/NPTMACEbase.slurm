#!/bin/bash -l
#SBATCH --job-name=NPT_MACE_base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load anaconda/3/2023.03

source /ptmp/kshao/envs/mlip_clean/bin/activate
export PYTHONNOUSERSITE=1
export PATH=/ptmp/kshao/mlip/packmol:$PATH


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun python NPTMACEbase.py
