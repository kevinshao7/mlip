#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIP_DIR="${MLIP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

bash "$MLIP_DIR/codes/A_parityplot/8_4_stormy/run_stormy.sh"
bash "$MLIP_DIR/codes/A_parityplot/8_5_dart9/run_dart9.sh"
bash "$MLIP_DIR/codes/A_parityplot/8_6_dart10/run_dart10.sh"
bash "$MLIP_DIR/codes/A_parityplot/8_7_dart11/run_dart11.sh"
