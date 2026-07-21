#!/bin/bash -l
#SBATCH --job-name=__JOB_NAME__
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%j.err

set -e

module purge
module load orca

OUTPUT_DIR=/ptmp/kshao/mlip/outputsfull/7_7c_orcaclusters
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
orca /ptmp/kshao/mlip/codes/7_7c_orcaclusters/expand/__INPUT_FILE__ > "$OUTPUT_DIR/__OUTPUT_FILE__"
