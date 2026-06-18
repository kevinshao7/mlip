#!/bin/bash -l
#SBATCH --job-name=orcadft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2023.03
module load git/2.50
module load cuda/12.8
module load orca

orca orcahellowater.inp > orcahellowater.out
