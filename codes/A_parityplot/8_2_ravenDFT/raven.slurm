#!/bin/bash -l
#SBATCH --job-name=make_isolatedH_raven
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.err

set -euo pipefail

module purge
module load orca/6.1.1

MLIP_DIR=${MLIP_DIR:-/ptmp/kshao/mlip}
FRAMES=${FRAMES:-0,100}
STARTUP="$MLIP_DIR/codes/A_parityplot/startup.py"

python "$STARTUP" --machine raven --frames "$FRAMES" --generate-only
