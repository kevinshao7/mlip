#!/bin/bash -l
#SBATCH --job-name=orca_E0_H
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.err

set -e

module purge
module load orca

mkdir -p /ptmp/kshao/mlip/outputsfull/7_4_orcaatomizations
cd /ptmp/kshao/mlip/outputsfull/7_4_orcaatomizations
orca /ptmp/kshao/mlip/codes/7_4_orcaatomizations/orcaatomizationH.inp > orcaatomizationH.out
