#!/usr/bin/env bash
set -euo pipefail

MLIP_DIR="${MLIP_DIR:-/ptmp/kshao/mlip}"
INPUT_DIR="$MLIP_DIR/codes/A_parityplot/8_5_dart9"
OUTPUT_DIR="$MLIP_DIR/outputsfull/A_parityplot/8_5_dart9"
ORCA_COMMAND=orca_qc
FORCE="${FORCE:-0}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

STEMS=(
    r09_hot_w_isolatedH_dart9_045
    r09_hot_w_isolatedH_dart9_046
    r09_hot_w_isolatedH_dart9_047
    r09_hot_w_isolatedH_dart9_048
    r09_hot_w_isolatedH_dart9_049
    r09_hot_w_isolatedH_dart9_050
    r09_hot_w_isolatedH_dart9_051
    r09_hot_w_isolatedH_dart9_052
    r09_hot_w_isolatedH_dart9_053
    r09_hot_w_isolatedH_dart9_054
    r09_hot_w_isolatedH_dart9_055
    r09_hot_w_isolatedH_dart9_056
    r09_hot_w_isolatedH_dart9_057
    r09_hot_w_isolatedH_dart9_058
    r09_hot_w_isolatedH_dart9_059
    r09_hot_w_isolatedH_dart9_060
    r09_hot_w_isolatedH_dart9_061
    r09_hot_w_isolatedH_dart9_062
    r09_hot_w_isolatedH_dart9_063
    r09_hot_w_isolatedH_dart9_064
    r09_hot_w_isolatedH_dart9_065
    r09_hot_w_isolatedH_dart9_066
    r09_hot_w_isolatedH_dart9_067
    r09_hot_w_isolatedH_dart9_068
    r09_hot_w_isolatedH_dart9_069
    r09_hot_w_isolatedH_dart9_070
    r09_hot_w_isolatedH_dart9_071
    r09_hot_w_isolatedH_dart9_072
    r09_hot_w_isolatedH_dart9_073
    r09_hot_w_isolatedH_dart9_074
    r09_hot_w_isolatedH_dart9_075
    r09_hot_w_isolatedH_dart9_076
    r09_hot_w_isolatedH_dart9_077
    r09_hot_w_isolatedH_dart9_078
    r09_hot_w_isolatedH_dart9_079
    r09_hot_w_isolatedH_dart9_080
    r09_hot_w_isolatedH_dart9_081
    r09_hot_w_isolatedH_dart9_082
    r09_hot_w_isolatedH_dart9_083
    r09_hot_w_isolatedH_dart9_084
    r09_hot_w_isolatedH_dart9_085
    r09_hot_w_isolatedH_dart9_086
    r09_hot_w_isolatedH_dart9_087
    r09_hot_w_isolatedH_dart9_088
    r09_hot_w_isolatedH_dart9_089
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
