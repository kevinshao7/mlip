#!/bin/bash
#SBATCH --job-name=r09_hot_w_isolatedH_bluehive_001
#SBATCH --partition=standard
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH -n 24
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH --time=12:00:00
#SBATCH -o r09_hot_w_isolatedH_bluehive_001.slurmlog.txt
#SBATCH -e r09_hot_w_isolatedH_bluehive_001.slurmerr.txt

set -euo pipefail

module purge
module load orca/6.1.1

MLIP_DIR="${MLIP_DIR:-/gpfs/fs2/scratch/kshao4/mlip}"
INPUT_PATH="$MLIP_DIR/codes/A_parityplot/8_5_bluehiveDFT/r09_hot_w_isolatedH_bluehive_001.inp"
OUTPUT_DIR="$MLIP_DIR/outputsfull/A_parityplot/8_5_bluehiveDFT"
OUTPUT_PATH="$OUTPUT_DIR/r09_hot_w_isolatedH_bluehive_001.out"
ORCA_COMMAND='orca'
ORCA_COMMAND="$(command -v "$ORCA_COMMAND")"

echo "MLIP_DIR=$MLIP_DIR"
echo "ORCA_COMMAND=$ORCA_COMMAND"
echo "INPUT_PATH=$INPUT_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"

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
$ORCA_COMMAND "$INPUT_PATH" > "$OUTPUT_PATH"

grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH"
grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"
