#!/usr/bin/env bash
set -euo pipefail
MLIP_DIR="${MLIP_DIR:-/ptmp/kshao/mlip}"

bash "$MLIP_DIR/codes/A_parityplot/8_4_stormy/run_stormy.sh"
bash "$MLIP_DIR/codes/A_parityplot/8_5_dart9/run_dart9.sh"
bash "$MLIP_DIR/codes/A_parityplot/8_6_dart10/run_dart10.sh"
bash "$MLIP_DIR/codes/A_parityplot/8_7_dart11/run_dart11.sh"
