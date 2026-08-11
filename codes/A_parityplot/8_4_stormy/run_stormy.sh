#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIP_DIR="${MLIP_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INPUT_DIR="$MLIP_DIR/codes/A_parityplot/8_4_stormy"
OUTPUT_DIR="$MLIP_DIR/outputsfull/A_parityplot/8_4_stormy"
ORCA_COMMAND=orca_qc
BASIS_FILE="def2-tzvpd.bas"
FORCE="${FORCE:-1}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

if ! type module >/dev/null 2>&1; then
    [[ -f /etc/profile ]] && source /etc/profile
    [[ -f "$HOME/.bashrc" ]] && source "$HOME/.bashrc"
    [[ -f "$HOME/.bash_profile" ]] && source "$HOME/.bash_profile"
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
fi
if type module >/dev/null 2>&1; then
    module load mpi/mpich-x86_64
else
    echo "module command not found; skipping module load mpi/mpich-x86_64" >&2
fi
if [[ -n "${MPI_BIN_DIR:-}" ]]; then
    export PATH="$MPI_BIN_DIR:$PATH"
fi
if ! command -v mpirun >/dev/null 2>&1; then
    for candidate in /usr/lib64/mpich/bin /usr/lib/x86_64-linux-gnu/mpich/bin /usr/local/mpich/bin /opt/mpich/bin /usr/lib64/openmpi/bin /usr/lib/x86_64-linux-gnu/openmpi/bin /usr/local/openmpi/bin /opt/openmpi/bin; do
        if [[ -x "$candidate/mpirun" ]]; then
            export PATH="$candidate:$PATH"
            break
        fi
    done
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "mpirun not found. Load mpi/mpich-x86_64 or set MPI_BIN_DIR=/path/to/mpi/bin before running this script." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
if [[ -f "$INPUT_DIR/$BASIS_FILE" ]]; then
    cp "$INPUT_DIR/$BASIS_FILE" "$OUTPUT_DIR/$BASIS_FILE"
else
    echo "Missing ORCA basis file: $INPUT_DIR/$BASIS_FILE" >&2
    exit 1
fi
cd "$OUTPUT_DIR"

STEMS=(
    r09_hot_w_isolatedH_stormy_000
    r09_hot_w_isolatedH_stormy_001
    r09_hot_w_isolatedH_stormy_002
    r09_hot_w_isolatedH_stormy_003
    r09_hot_w_isolatedH_stormy_004
    r09_hot_w_isolatedH_stormy_005
    r09_hot_w_isolatedH_stormy_006
    r09_hot_w_isolatedH_stormy_007
    r09_hot_w_isolatedH_stormy_008
    r09_hot_w_isolatedH_stormy_009
    r09_hot_w_isolatedH_stormy_010
    r09_hot_w_isolatedH_stormy_011
    r09_hot_w_isolatedH_stormy_012
    r09_hot_w_isolatedH_stormy_013
    r09_hot_w_isolatedH_stormy_014
    r09_hot_w_isolatedH_stormy_015
    r09_hot_w_isolatedH_stormy_016
    r09_hot_w_isolatedH_stormy_017
    r09_hot_w_isolatedH_stormy_018
    r09_hot_w_isolatedH_stormy_019
    r09_hot_w_isolatedH_stormy_020
    r09_hot_w_isolatedH_stormy_021
    r09_hot_w_isolatedH_stormy_022
    r09_hot_w_isolatedH_stormy_023
    r09_hot_w_isolatedH_stormy_024
    r09_hot_w_isolatedH_stormy_025
    r09_hot_w_isolatedH_stormy_026
    r09_hot_w_isolatedH_stormy_027
    r09_hot_w_isolatedH_stormy_028
    r09_hot_w_isolatedH_stormy_029
    r09_hot_w_isolatedH_stormy_030
    r09_hot_w_isolatedH_stormy_031
    r09_hot_w_isolatedH_stormy_032
    r09_hot_w_isolatedH_stormy_033
    r09_hot_w_isolatedH_stormy_034
    r09_hot_w_isolatedH_stormy_035
    r09_hot_w_isolatedH_stormy_036
    r09_hot_w_isolatedH_stormy_037
    r09_hot_w_isolatedH_stormy_038
    r09_hot_w_isolatedH_stormy_039
    r09_hot_w_isolatedH_stormy_040
    r09_hot_w_isolatedH_stormy_041
    r09_hot_w_isolatedH_stormy_042
    r09_hot_w_isolatedH_stormy_043
    r09_hot_w_isolatedH_stormy_044
)

for STEM in "${STEMS[@]}"; do
    INPUT_PATH="$INPUT_DIR/$STEM.inp"
    OUTPUT_PATH="$OUTPUT_DIR/$STEM.out"

    if [[ ! -f "$INPUT_PATH" ]]; then
        echo "Missing ORCA input: $INPUT_PATH" >&2
        exit 1
    fi

    if [[ -f "$OUTPUT_PATH" ]] && grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH" && grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"; then
        echo "Skipping completed $OUTPUT_PATH"
        continue
    fi

    if [[ -f "$OUTPUT_PATH" ]]; then
        if [[ "$FORCE" == "1" ]]; then
            rm -f "$OUTPUT_PATH"
        else
            echo "Output already exists or is incomplete: $OUTPUT_PATH" >&2
            echo "Use FORCE=1 bash $(basename "$0") to overwrite incomplete output." >&2
            exit 1
        fi
    fi

    echo "Running $INPUT_PATH -> $OUTPUT_PATH"
    $ORCA_COMMAND "$INPUT_PATH" > "$OUTPUT_PATH"

    grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH"
    grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"
done
