#!/bin/bash -l
#SBATCH --job-name=r09_hot_w_isolatedH_greatlakes_103
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/kevinsh/mlip/outputsfull/slurm/r09_hot_w_isolatedH_greatlakes_103-%j.out
#SBATCH --error=/home/kevinsh/mlip/outputsfull/slurm/r09_hot_w_isolatedH_greatlakes_103-%j.err

set -euo pipefail

module purge
module load orca/6.0.1

MLIP_DIR=${MLIP_DIR:-/home/kevinsh/mlip}
INPUT_PATH="$MLIP_DIR/codes/A_parityplot/8_4_greatlakesDFT/r09_hot_w_isolatedH_greatlakes_103.inp"
OUTPUT_DIR="$MLIP_DIR/outputsfull/A_parityplot/8_4_greatlakesDFT"
OUTPUT_PATH="$OUTPUT_DIR/r09_hot_w_isolatedH_greatlakes_103.out"
ORCA_COMMAND='orca'

mkdir -p "$MLIP_DIR/outputsfull/slurm"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Missing ORCA input: $INPUT_PATH" >&2
    exit 1
fi

if [[ -f "$OUTPUT_PATH" ]] && grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH" && grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"; then
    echo "Skipping completed $OUTPUT_PATH"
    exit 0
fi

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

cd "$OUTPUT_DIR"
$ORCA_COMMAND "$INPUT_PATH" > "$OUTPUT_PATH"

grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH"
grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"
