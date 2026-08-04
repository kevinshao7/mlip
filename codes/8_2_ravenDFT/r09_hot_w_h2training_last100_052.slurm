#!/bin/bash -l
#SBATCH --job-name=orca_r09_hot_w_h2training_last100_052
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.err

set -e

module purge
module load orca

OUTPUT_DIR=/ptmp/kshao/mlip/outputsfull/8_2_ravenDFT
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
orca /ptmp/kshao/mlip/codes/8_2_ravenDFT/r09_hot_w_h2training_last100_052.inp > "$OUTPUT_DIR/r09_hot_w_h2training_last100_052.out"
