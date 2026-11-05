#!/bin/bash
set -euo pipefail
workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$workflow_dir/../.." && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "${PYTHON_COMMAND:-$repo_dir/.venv/bin/python}" "$workflow_dir/dft/viper_cpu.py" submit "$@"
