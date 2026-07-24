#!/bin/bash -l
#SBATCH --job-name=daisjaxtoppot
#SBATCH --nodes=1

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1000000
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source /dais/fs/scratch/kshao/jaxmd-cli/.venv/bin/activate
cd /dais/fs/scratch/kshao/mlip
jaxmd-run /dais/fs/scratch/kshao/mlip/jaxcodes/harryrepex.toml
