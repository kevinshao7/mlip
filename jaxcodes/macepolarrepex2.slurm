#!/bin/bash -l
# IMPORTANT: MACE-POLAR requires the dev branch of external jax-md-cli.
#SBATCH --job-name=macepolarrepex
#SBATCH --nodes=1

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1000000
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

JAXMD_CLI_ROOT=/dais/fs/scratch/kshao/jaxmd-cli
cd "$JAXMD_CLI_ROOT"

source "$JAXMD_CLI_ROOT/.venv/bin/activate"
jaxmd-run /dais/fs/scratch/kshao/jaxmd-cli/example_configs/mace_polar_repex.toml

