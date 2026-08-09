#!/bin/bash -l
#SBATCH --job-name=isolatedH_viper
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%A_%a.out
#SBATCH --error=/ptmp/kshao/mlip/outputsfull/slurm/slurm-%x-%A_%a.err

set -euo pipefail

module purge
module load orca/6.1.1

MLIP_DIR=${MLIP_DIR:-/ptmp/kshao/mlip}
FRAME=${SLURM_ARRAY_TASK_ID:-${FRAME:-0}}
STEM=$(printf "r09_hot_w_isolatedH_viper_%03d" "$FRAME")
INPUT_PATH="$MLIP_DIR/codes/A_parityplot/8_1_viperDFT/$STEM.inp"
OUTPUT_DIR="$MLIP_DIR/outputsfull/A_parityplot/8_1_viperDFT"
OUTPUT_PATH="$OUTPUT_DIR/$STEM.out"

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

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

cd "$OUTPUT_DIR"
orca "$INPUT_PATH" > "$OUTPUT_PATH"

grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH"
grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"
