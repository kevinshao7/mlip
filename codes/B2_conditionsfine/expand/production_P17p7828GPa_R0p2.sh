#!/bin/bash -l

#SBATCH --job-name=fine_P17p7828GPa_R0p2
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=250000
#SBATCH --time=24:00:00

#SBATCH --chdir=/dais/fs/scratch/kshao/mlip/codes/B2_conditionsfine/expand
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
export MLIP_MACE_DEVICE=${MLIP_MACE_DEVICE:-cuda}
export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
export MLIP_ENV=${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}
export PYTHONNOUSERSITE=1
export PATH="$MLIP_ROOT/packmol:$PATH"
export XDG_CACHE_HOME="$MLIP_ROOT/outputsfull/.cache"

PYTHON_SCRIPT=production_P17p7828GPa_R0p2.py

mkdir -p "$MLIP_ROOT/outputsfull/slurm" "$XDG_CACHE_HOME"

echo "Job ID:        $SLURM_JOB_ID"
echo "Host:          $(hostname)"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "MPI tasks:     ${SLURM_NTASKS:-1}"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
echo "Visible GPU:   ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Python script: $PYTHON_SCRIPT"
echo "MLIP root:     $MLIP_ROOT"
echo "MLIP env:      $MLIP_ENV"
echo "MACE device:   $MLIP_MACE_DEVICE"

module list 2>&1
nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv

test -f "$PYTHON_SCRIPT"
test -f "$MLIP_ENV/bin/activate"
command -v packmol

source "$MLIP_ENV/bin/activate"
python -c "import ase, mace, mdinterface, torch, graph_longrange; print('Python env OK')"
python -c "import shutil; exe=shutil.which('packmol'); print(f'Packmol OK: {exe}'); assert exe"

srun \
    --ntasks="${SLURM_NTASKS:-1}" \
    --kill-on-bad-exit=1 \
    python "$PYTHON_SCRIPT"

echo "MD completed successfully."
echo "End time: $(date)"
