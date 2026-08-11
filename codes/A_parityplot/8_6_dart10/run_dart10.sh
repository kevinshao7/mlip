#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIP_DIR="${MLIP_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INPUT_DIR="$MLIP_DIR/codes/A_parityplot/8_6_dart10"
OUTPUT_DIR="$MLIP_DIR/outputsfull/A_parityplot/8_6_dart10"
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
    r09_hot_w_isolatedH_dart10_090
    r09_hot_w_isolatedH_dart10_091
    r09_hot_w_isolatedH_dart10_092
    r09_hot_w_isolatedH_dart10_093
    r09_hot_w_isolatedH_dart10_094
    r09_hot_w_isolatedH_dart10_095
    r09_hot_w_isolatedH_dart10_096
    r09_hot_w_isolatedH_dart10_097
    r09_hot_w_isolatedH_dart10_098
    r09_hot_w_isolatedH_dart10_099
    r09_hot_w_isolatedH_dart10_100
    r09_hot_w_isolatedH_dart10_101
    r09_hot_w_isolatedH_dart10_102
    r09_hot_w_isolatedH_dart10_103
    r09_hot_w_isolatedH_dart10_104
    r09_hot_w_isolatedH_dart10_105
    r09_hot_w_isolatedH_dart10_106
    r09_hot_w_isolatedH_dart10_107
    r09_hot_w_isolatedH_dart10_108
    r09_hot_w_isolatedH_dart10_109
    r09_hot_w_isolatedH_dart10_110
    r09_hot_w_isolatedH_dart10_111
    r09_hot_w_isolatedH_dart10_112
    r09_hot_w_isolatedH_dart10_113
    r09_hot_w_isolatedH_dart10_114
    r09_hot_w_isolatedH_dart10_115
    r09_hot_w_isolatedH_dart10_116
    r09_hot_w_isolatedH_dart10_117
    r09_hot_w_isolatedH_dart10_118
    r09_hot_w_isolatedH_dart10_119
    r09_hot_w_isolatedH_dart10_120
    r09_hot_w_isolatedH_dart10_121
    r09_hot_w_isolatedH_dart10_122
    r09_hot_w_isolatedH_dart10_123
    r09_hot_w_isolatedH_dart10_124
    r09_hot_w_isolatedH_dart10_125
    r09_hot_w_isolatedH_dart10_126
    r09_hot_w_isolatedH_dart10_127
    r09_hot_w_isolatedH_dart10_128
    r09_hot_w_isolatedH_dart10_129
    r09_hot_w_isolatedH_dart10_130
    r09_hot_w_isolatedH_dart10_131
    r09_hot_w_isolatedH_dart10_132
    r09_hot_w_isolatedH_dart10_133
    r09_hot_w_isolatedH_dart10_134
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
