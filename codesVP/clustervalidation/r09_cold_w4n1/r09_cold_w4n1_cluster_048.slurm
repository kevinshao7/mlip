#!/bin/bash -l
#SBATCH --job-name=r09_cold_w4n1_cluster_048
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load orca

orca r09_cold_w4n1_cluster_048.inp > r09_cold_w4n1_cluster_048.out
