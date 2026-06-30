#!/bin/bash -l
#SBATCH --job-name=orca_E0_O
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load orca

/mpcdf/soft/RHEL_9/packages/x86_64/orca/6.1.1/bin/orca /ptmp/kshao/mlip/codesVP/orcaatomizations/orcaatomizationO.inp > /ptmp/kshao/mlip/codesVP/orcaatomizations/expandorcaatomizationO.out
