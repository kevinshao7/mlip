#!/bin/bash
#SBATCH --job-name=C_DFTprod_cutcluster_1987
#SBATCH --partition=standard
#SBATCH --mail-type=END
#SBATCH --mail-user=ks2120@cam.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -n 24
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH --time=12:00:00
#SBATCH -o C_DFTprod_cutcluster_1987.slurmlog.txt
#SBATCH -e C_DFTprod_cutcluster_1987.slurmerr.txt

set -euo pipefail

module purge
module load orca/6.1.1

MLIP_DIR="${MLIP_DIR:-/gpfs/fs2/scratch/kshao4/mlip}"
INPUT_PATH="$MLIP_DIR/codes/C_DFTproduction/expand/C_DFTprod_cutcluster_1987.inp"
INPUT_DIR="$MLIP_DIR/codes/C_DFTproduction"
OUTPUT_DIR="$MLIP_DIR/outputsfull/C_DFTproduction/dft_outputs"
OUTPUT_PATH="$OUTPUT_DIR/C_DFTprod_cutcluster_1987.out"
BASIS_PATH="$INPUT_DIR/def2-tzvpd.bas"
ORCA_COMMAND="${ORCA_COMMAND:-/software/orca/6.1.1/orca}"

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
echo "INPUT_PATH=$INPUT_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "mpirun=$(command -v mpirun)"

mkdir -p "$MLIP_DIR/outputsfull/slurm"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Missing ORCA input: $INPUT_PATH" >&2
    exit 1
fi
if [[ ! -f "$BASIS_PATH" ]]; then
    echo "Missing ORCA basis file: $BASIS_PATH" >&2
    exit 1
fi
if [[ ! -f "$OUTPUT_DIR/def2-tzvpd.bas" ]]; then
    cp "$BASIS_PATH" "$OUTPUT_DIR/def2-tzvpd.bas"
fi

if [[ -f "$OUTPUT_PATH" ]] && grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH" && grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"; then
    echo "Skipping completed $OUTPUT_PATH"
    exit 0
fi

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export OMPI_MCA_btl=^openib
export OMPI_MCA_orte_base_help_aggregate=0

cd "$OUTPUT_DIR"
"$ORCA_COMMAND" "$INPUT_PATH" > "$OUTPUT_PATH"

grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH"
grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"
