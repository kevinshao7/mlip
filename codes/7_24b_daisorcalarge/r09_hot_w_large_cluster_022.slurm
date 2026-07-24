#!/bin/bash -l
#SBATCH --job-name=orca_r09_hot_w_large_cluster_022
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.err

set -e

module purge
module load orca

OUTPUT_DIR=/ptmp/kshao/mlip/outputsfull/7_24b_daisorcalarge
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
orca /ptmp/kshao/mlip/codes/7_24b_daisorcalarge/r09_hot_w_large_cluster_022.inp > "$OUTPUT_DIR/r09_hot_w_large_cluster_022.out"
