#!/bin/bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_COMMAND:-/viper/ptmp1/kshao/mlip/.venv/bin/python}" /viper/ptmp1/kshao/mlip/codes/X_semiactivelearning/dft/viper_cpu.py --package "$bundle_dir/package.json" submit "$@"
