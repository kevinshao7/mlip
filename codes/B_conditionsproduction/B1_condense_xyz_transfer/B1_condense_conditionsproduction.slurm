#!/bin/bash -l

#SBATCH --job-name=B1_condense_xyz
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000
#SBATCH --time=24:00:00

#SBATCH --chdir=/dais/fs/scratch/kshao/mlip/codes/B_conditionsproduction/B1_condense_xyz_transfer
#SBATCH --output=/dais/fs/scratch/kshao/mlip/outputsfull/slurm/%x_%j.out
#SBATCH --error=/dais/fs/scratch/kshao/mlip/outputsfull/slurm/%x_%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ks2120@cam.ac.uk

set -euo pipefail

module purge
unset MPI_ROOT
unset CUDA_ROOT
unset LD_LIBRARY_PATH

module load gcc/14
module load cuda/12.8

export OMP_NUM_THREADS=1
export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
export MLIP_ENV=${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$MLIP_ROOT/outputsfull/.cache"

PYTHON_SCRIPT=condense_xyz_stride.py
INPUT_ROOT=${INPUT_ROOT:-$MLIP_ROOT/outputsfull/conditionsproduction}
OUTPUT_ROOT=${OUTPUT_ROOT:-$MLIP_ROOT/outputsfull/B1_conditionsproduction_stride100_xyz}
STRIDE=${STRIDE:-100}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-12}}

mkdir -p "$MLIP_ROOT/outputsfull/slurm" "$XDG_CACHE_HOME" "$OUTPUT_ROOT"

echo "Job ID:        $SLURM_JOB_ID"
echo "Host:          $(hostname)"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
echo "Visible GPU:   ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Python script: $PYTHON_SCRIPT"
echo "MLIP root:     $MLIP_ROOT"
echo "Input root:    $INPUT_ROOT"
echo "Output root:   $OUTPUT_ROOT"
echo "Stride:        $STRIDE"
echo "Workers:       $WORKERS"

module list 2>&1
nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

test -f "$PYTHON_SCRIPT"

if [[ -f "$MLIP_ENV/bin/activate" ]]; then
    source "$MLIP_ENV/bin/activate"
fi

python -c "print('Python OK')"

srun \
    --ntasks="${SLURM_NTASKS:-1}" \
    --kill-on-bad-exit=1 \
    python "$PYTHON_SCRIPT" \
        --input-root "$INPUT_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --stride "$STRIDE" \
        --workers "$WORKERS"

echo "Condense completed successfully."
echo "End time: $(date)"
