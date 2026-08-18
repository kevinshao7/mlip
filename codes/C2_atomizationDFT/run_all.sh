#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCA_COMMAND="${ORCA_COMMAND:-orca_qc}"
MPI_BIN_DIR="${MPI_BIN_DIR:-/usr/lib64/openmpi/bin}"
MPI_LIB_DIR="${MPI_LIB_DIR:-/usr/lib64/openmpi/lib}"
ATOMS=(H O N S)

if [[ -d "$MPI_BIN_DIR" ]]; then
    export PATH="$MPI_BIN_DIR:$PATH"
fi
if [[ -d "$MPI_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

if ! command -v "$ORCA_COMMAND" >/dev/null 2>&1; then
    echo "ORCA command is not available: $ORCA_COMMAND" >&2
    exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "mpirun is not available; checked PATH and MPI_BIN_DIR=$MPI_BIN_DIR" >&2
    exit 1
fi

for atom in "${ATOMS[@]}"; do
    run_dir="$SCRIPT_DIR/runs/$atom"
    stem="orcaatomization${atom}"
    input_path="$run_dir/${stem}.inp"
    output_path="$run_dir/${stem}.out"

    if [[ ! -f "$input_path" ]]; then
        echo "Missing ORCA input: $input_path" >&2
        exit 1
    fi
    if [[ ! -f "$run_dir/def2-tzvpd.bas" ]]; then
        echo "Missing basis file in run directory: $run_dir/def2-tzvpd.bas" >&2
        exit 1
    fi

    echo "Running $stem with ORCA command: $ORCA_COMMAND"
    echo "Input: $input_path"
    echo "Output: $output_path"
    (
        cd "$run_dir"
        "$ORCA_COMMAND" "${stem}.inp" | tee "${stem}.out"
    )
    grep -q "FINAL SINGLE POINT ENERGY" "$output_path"
    grep -q "ORCA TERMINATED NORMALLY" "$output_path"
    echo "Finished $stem"
done
