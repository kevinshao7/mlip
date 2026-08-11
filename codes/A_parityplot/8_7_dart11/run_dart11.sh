#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIP_DIR="${MLIP_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INPUT_DIR="$MLIP_DIR/codes/A_parityplot/8_7_dart11"
OUTPUT_DIR="$MLIP_DIR/outputsfull/A_parityplot/8_7_dart11"
ORCA_COMMAND=orca_qc
BASIS_FILE="def2-tzvpd.bas"
FORCE="${FORCE:-1}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

mkdir -p "$OUTPUT_DIR"
if [[ -f "$INPUT_DIR/$BASIS_FILE" ]]; then
    cp "$INPUT_DIR/$BASIS_FILE" "$OUTPUT_DIR/$BASIS_FILE"
else
    echo "Missing ORCA basis file: $INPUT_DIR/$BASIS_FILE" >&2
    exit 1
fi
cd "$OUTPUT_DIR"

STEMS=(
    r09_hot_w_isolatedH_dart11_135
    r09_hot_w_isolatedH_dart11_136
    r09_hot_w_isolatedH_dart11_137
    r09_hot_w_isolatedH_dart11_138
    r09_hot_w_isolatedH_dart11_139
    r09_hot_w_isolatedH_dart11_140
    r09_hot_w_isolatedH_dart11_141
    r09_hot_w_isolatedH_dart11_142
    r09_hot_w_isolatedH_dart11_143
    r09_hot_w_isolatedH_dart11_144
    r09_hot_w_isolatedH_dart11_145
    r09_hot_w_isolatedH_dart11_146
    r09_hot_w_isolatedH_dart11_147
    r09_hot_w_isolatedH_dart11_148
    r09_hot_w_isolatedH_dart11_149
    r09_hot_w_isolatedH_dart11_150
    r09_hot_w_isolatedH_dart11_151
    r09_hot_w_isolatedH_dart11_152
    r09_hot_w_isolatedH_dart11_153
    r09_hot_w_isolatedH_dart11_154
    r09_hot_w_isolatedH_dart11_155
    r09_hot_w_isolatedH_dart11_156
    r09_hot_w_isolatedH_dart11_157
    r09_hot_w_isolatedH_dart11_158
    r09_hot_w_isolatedH_dart11_159
    r09_hot_w_isolatedH_dart11_160
    r09_hot_w_isolatedH_dart11_161
    r09_hot_w_isolatedH_dart11_162
    r09_hot_w_isolatedH_dart11_163
    r09_hot_w_isolatedH_dart11_164
    r09_hot_w_isolatedH_dart11_165
    r09_hot_w_isolatedH_dart11_166
    r09_hot_w_isolatedH_dart11_167
    r09_hot_w_isolatedH_dart11_168
    r09_hot_w_isolatedH_dart11_169
    r09_hot_w_isolatedH_dart11_170
    r09_hot_w_isolatedH_dart11_171
    r09_hot_w_isolatedH_dart11_172
    r09_hot_w_isolatedH_dart11_173
    r09_hot_w_isolatedH_dart11_174
    r09_hot_w_isolatedH_dart11_175
    r09_hot_w_isolatedH_dart11_176
    r09_hot_w_isolatedH_dart11_177
    r09_hot_w_isolatedH_dart11_178
    r09_hot_w_isolatedH_dart11_179
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
