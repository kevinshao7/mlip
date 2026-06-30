#!/bin/bash -l
#SBATCH --job-name=orca_E0_N
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load orca

orca /ptmp/kshao/mlip/codesVP/orcaatomizations/orcaatomizationN.inp > /ptmp/kshao/mlip/codesVP/orcaatomizations/expandorcaatomizationN.out
