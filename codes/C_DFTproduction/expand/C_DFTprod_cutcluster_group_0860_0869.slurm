#!/bin/bash
#SBATCH --job-name=C_DFTprod_cutcluster_group_0860_0869
#SBATCH --partition=standard
# Email disabled for middle jobs to avoid notification spam.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -n 24
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH --time=120:00:00
#SBATCH -o C_DFTprod_cutcluster_group_0860_0869.slurmlog.txt
#SBATCH -e C_DFTprod_cutcluster_group_0860_0869.slurmerr.txt

set -euo pipefail

module purge
module load orca/6.1.1

MLIP_DIR="${MLIP_DIR:-/gpfs/fs2/scratch/kshao4/mlip}"
INPUT_DIR="$MLIP_DIR/codes/C_DFTproduction"
OUTPUT_DIR="$MLIP_DIR/outputsfull/C_DFTproduction/dft_outputs"
BASIS_PATH="$INPUT_DIR/def2-tzvpd.bas"
ORCA_COMMAND="${ORCA_COMMAND:-/software/orca/6.1.1/orca}"
STEMS=(
    "C_DFTprod_cutcluster_0860"
    "C_DFTprod_cutcluster_0861"
    "C_DFTprod_cutcluster_0862"
    "C_DFTprod_cutcluster_0863"
    "C_DFTprod_cutcluster_0864"
    "C_DFTprod_cutcluster_0865"
    "C_DFTprod_cutcluster_0866"
    "C_DFTprod_cutcluster_0867"
    "C_DFTprod_cutcluster_0868"
    "C_DFTprod_cutcluster_0869"
)

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
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "mpirun=$(command -v mpirun)"
echo "Running ${#STEMS[@]} sequential ORCA calculation(s)"

mkdir -p "$MLIP_DIR/outputsfull/slurm"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$BASIS_PATH" ]]; then
    echo "Missing ORCA basis file: $BASIS_PATH" >&2
    exit 1
fi
if [[ ! -f "$OUTPUT_DIR/def2-tzvpd.bas" ]]; then
    cp "$BASIS_PATH" "$OUTPUT_DIR/def2-tzvpd.bas"
fi

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export OMPI_MCA_btl=^openib
export OMPI_MCA_orte_base_help_aggregate=0

cd "$OUTPUT_DIR"

for STEM in "${STEMS[@]}"; do
    INPUT_PATH="$MLIP_DIR/codes/C_DFTproduction/expand/${STEM}.inp"
    OUTPUT_PATH="$OUTPUT_DIR/${STEM}.out"

    if [[ ! -f "$INPUT_PATH" ]]; then
        echo "Missing ORCA input: $INPUT_PATH" >&2
        exit 1
    fi

    if [[ -f "$OUTPUT_PATH" ]] && grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH" && grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"; then
        echo "Skipping completed $OUTPUT_PATH"
        continue
    fi

    echo "Starting $STEM"
    "$ORCA_COMMAND" "$INPUT_PATH" > "$OUTPUT_PATH"
    grep -q "FINAL SINGLE POINT ENERGY" "$OUTPUT_PATH"
    grep -q "ORCA TERMINATED NORMALLY" "$OUTPUT_PATH"
    echo "Finished $STEM"
done
