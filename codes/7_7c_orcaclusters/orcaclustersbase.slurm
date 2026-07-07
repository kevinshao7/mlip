#!/bin/bash -l
#SBATCH --job-name=orca_E0_S
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load orca

cd /ptmp/kshao/mlip/codes/7_7c_orcaclusters
orca orcaatomizationS.inp > orcaatomizationS.out
