#!/bin/bash -l
#SBATCH --job-name=orca_E0_S
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load orca

C:\ORCA_6.1.1\orca.exe C:\Users\shaoq\Documents\Mainz\mlip\codesPC\orcaatomizations\orcaatomizationS.inp > C:\Users\shaoq\Documents\Mainz\mlip\codesPC\orcaatomizations\orcaatomizationS.out
