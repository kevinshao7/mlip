#!/bin/bash
#SBATCH --job-name=C3_DFTprod_stopH2_group_0060_0069
#SBATCH --partition=standard
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -n 24
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH --time=120:00:00
#SBATCH -o C3_DFTprod_stopH2_group_0060_0069.slurmlog.txt
#SBATCH -e C3_DFTprod_stopH2_group_0060_0069.slurmerr.txt

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
OUTPUT_DIR="$MLIP_DIR/outputsfull/C3_DFTproductionstopH2_ON/dft_outputs"
BASIS_PATH="$INPUT_DIR/def2-tzvpd.bas"
ORCA_COMMAND="${ORCA_COMMAND:-/software/orca/6.1.1/orca}"
STEMS=(
    "C3_DFTprod_stopH2_0060"
    "C3_DFTprod_stopH2_0061"
    "C3_DFTprod_stopH2_0062"
    "C3_DFTprod_stopH2_0063"
    "C3_DFTprod_stopH2_0064"
    "C3_DFTprod_stopH2_0065"
    "C3_DFTprod_stopH2_0066"
    "C3_DFTprod_stopH2_0067"
    "C3_DFTprod_stopH2_0068"
    "C3_DFTprod_stopH2_0069"
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
    TEMP_OUTPUT_PATH="$OUTPUT_DIR/${STEM}.out.tmp.$$"

    if [[ ! -f "$INPUT_PATH" ]]; then
        echo "Missing ORCA input: $INPUT_PATH" >&2
        exit 1
    fi

    EXPECTED_CHARGE="$(awk 'tolower($1) == "*xyz" { print $2; exit }' "$INPUT_PATH")"
    if [[ -z "$EXPECTED_CHARGE" ]]; then
        echo "Could not read *xyz charge from $INPUT_PATH" >&2
        exit 1
    fi

    if [[ -f "$OUTPUT_PATH" ]] && grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH" && grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"; then
        # Use the first General Settings value. Never use a population-analysis
        # "total charge" or metadata from a newly regenerated input.
        OUTPUT_CHARGE="$(awk '/Total Charge[[:space:]]+Charge[[:space:]]+\.\.\.\./ { print $NF; exit }' "$OUTPUT_PATH")"
        if [[ "$OUTPUT_CHARGE" == "$EXPECTED_CHARGE" ]]; then
            echo "Skipping completed $OUTPUT_PATH (verified charge $OUTPUT_CHARGE)"
            continue
        fi
        echo "Recalculating $STEM: existing output charge ${OUTPUT_CHARGE:-unknown}, expected $EXPECTED_CHARGE"
    fi

    echo "Starting $STEM"
    # ORCA automatically restarts from same-basename files beside the input.
    # Regenerated inputs may have different geometries, so never reuse them implicitly.
    RESTART_BASE="${INPUT_PATH%.inp}"
    rm -f "${RESTART_BASE}.gbw" "${RESTART_BASE}.ges"
    rm -f "$TEMP_OUTPUT_PATH"
    if ! "$ORCA_COMMAND" "$INPUT_PATH" > "$TEMP_OUTPUT_PATH"; then
        rm -f "$TEMP_OUTPUT_PATH"
        echo "ORCA failed for $STEM; preserving any existing $OUTPUT_PATH" >&2
        exit 1
    fi
    grep -q "FINAL SINGLE POINT ENERGY" "$TEMP_OUTPUT_PATH"
    grep -q "ORCA TERMINATED NORMALLY" "$TEMP_OUTPUT_PATH"
    OUTPUT_CHARGE="$(awk '/Total Charge[[:space:]]+Charge[[:space:]]+\.\.\.\./ { print $NF; exit }' "$TEMP_OUTPUT_PATH")"
    if [[ "$OUTPUT_CHARGE" != "$EXPECTED_CHARGE" ]]; then
        rm -f "$TEMP_OUTPUT_PATH"
        echo "Charge verification failed for $STEM: ORCA reported ${OUTPUT_CHARGE:-unknown}, expected $EXPECTED_CHARGE" >&2
        exit 1
    fi
    mv -f "$TEMP_OUTPUT_PATH" "$OUTPUT_PATH"
    echo "Finished $STEM"
done
