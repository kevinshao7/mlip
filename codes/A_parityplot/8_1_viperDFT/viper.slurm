#!/bin/bash -l
#SBATCH --job-name=isolatedH_viper
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%A_%a.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%A_%a.err

set -euo pipefail

module purge
module load orca/6.1.1

MLIP_DIR=${MLIP_DIR:-/ptmp/kshao/mlip}
FRAMES=${FRAMES:-0,100}
STARTUP="$MLIP_DIR/codes/A_parityplot/8_3_dungeonDFT/startup.py"
FRAME_START=${FRAMES%%,*}
TASK_INDEX=$((FRAME_START + SLURM_ARRAY_TASK_ID))

python "$STARTUP" --machine viper --frames "$FRAMES" --task-index "$TASK_INDEX" --resume
