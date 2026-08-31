#!/bin/bash
#SBATCH --job-name=C3_DFTprod_stopH2_group_0970_0979
#SBATCH --partition=standard
# Email disabled for middle jobs to avoid notification spam.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -n 24
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH --time=120:00:00
#SBATCH -o C3_DFTprod_stopH2_group_0970_0979.slurmlog.txt
#SBATCH -e C3_DFTprod_stopH2_group_0970_0979.slurmerr.txt

set -euo pipefail

filter_module_stderr() {
    grep -v -F "unalias: sudo: not found" >&2 || true
}

set +e
module purge 2> >(filter_module_stderr)
module load orca/6.1.1 2> >(filter_module_stderr)
MODULE_LOAD_STATUS=$?
set -e
if [[ "$MODULE_LOAD_STATUS" -ne 0 ]]; then
    echo "Failed to load orca/6.1.1 module" >&2
    exit "$MODULE_LOAD_STATUS"
fi

MLIP_DIR="${MLIP_DIR:-/gpfs/fs2/scratch/kshao4/mlip}"
INPUT_DIR="$MLIP_DIR/codes/C3_DFTproductionstopH2"
OUTPUT_DIR="$MLIP_DIR/outputsfull/C3_DFTproductionstopH2/dft_outputs"
BASIS_PATH="$INPUT_DIR/def2-tzvpd.bas"
ORCA_COMMAND="${ORCA_COMMAND:-/software/orca/6.1.1/orca}"
STEMS=(
    "C3_DFTprod_stopH2_0970"
    "C3_DFTprod_stopH2_0971"
    "C3_DFTprod_stopH2_0972"
    "C3_DFTprod_stopH2_0973"
    "C3_DFTprod_stopH2_0974"
    "C3_DFTprod_stopH2_0975"
    "C3_DFTprod_stopH2_0976"
    "C3_DFTprod_stopH2_0977"
    "C3_DFTprod_stopH2_0978"
    "C3_DFTprod_stopH2_0979"
)

if [[ "$ORCA_COMMAND" != /* ]]; then
    ORCA_COMMAND="$(command -v "$ORCA_COMMAND")"
fi
if [[ -z "$ORCA_COMMAND" || ! -x "$ORCA_COMMAND" ]]; then
    echo "ORCA executable is not available: ${ORCA_COMMAND:-unset}" >&2
    exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "mpirun is not available after loading orca/6.1.1" >&2
    exit 1
fi

echo "MLIP_DIR=$MLIP_DIR"
echo "ORCA_COMMAND=$ORCA_COMMAND"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "mpirun=$(command -v mpirun)"
echo "Running ${#STEMS[@]} sequential ORCA calculation(s)"

mkdir -p "$MLIP_DIR/outputsfull/slurm"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$BASIS_PATH" ]]; then
    echo "Missing ORCA basis file: $BASIS_PATH" >&2
    exit 1
fi
TARGET_BASIS="$OUTPUT_DIR/def2-tzvpd.bas"
if [[ ! -f "$TARGET_BASIS" ]]; then
    TMP_BASIS="$OUTPUT_DIR/def2-tzvpd.bas.${SLURM_JOB_ID:-$$}.tmp"
    cp "$BASIS_PATH" "$TMP_BASIS"
    mv -n "$TMP_BASIS" "$TARGET_BASIS"
    rm -f "$TMP_BASIS"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMPI_MCA_btl=^openib
export OMPI_MCA_orte_base_help_aggregate=0

cd "$OUTPUT_DIR"

for STEM in "${STEMS[@]}"; do
    INPUT_PATH="$MLIP_DIR/codes/C3_DFTproductionstopH2/expand/${STEM}.inp"
    OUTPUT_PATH="$OUTPUT_DIR/${STEM}.out"

    if [[ ! -f "$INPUT_PATH" ]]; then
        echo "Missing ORCA input: $INPUT_PATH" >&2
        exit 1
    fi

    if [[ -f "$OUTPUT_PATH" ]] && grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH" && grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"; then
        echo "Skipping completed $OUTPUT_PATH"
        continue
    fi

    echo "Starting $STEM"
    "$ORCA_COMMAND" "$INPUT_PATH" > "$OUTPUT_PATH"
    grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH"
    grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"
    echo "Finished $STEM"
done
