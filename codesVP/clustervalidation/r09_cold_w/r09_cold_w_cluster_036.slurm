#!/bin/bash -l
#SBATCH --job-name=r09_cold_w_cluster_036
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load orca

orca r09_cold_w_cluster_036.inp > r09_cold_w_cluster_036.out
