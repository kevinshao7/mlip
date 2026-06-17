#!/bin/bash -l
#SBATCH --job-name=mace_polar_debug
#SBATCH --partition=gpu
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2023.03
module load git/2.50
module load cuda/12.8

source ../../env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_LAUNCH_BLOCKING=1

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

srun python train_mace.py \
  --name="polar_ft_1m_debug" \
  --model="PolarMACE" \
  --foundation_model="polar-1-s" \
  --train_file="../ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz" \
  --valid_fraction=0.05 \
  --energy_key="TotEnergy" \
  --forces_key="force" \
  --compute_forces True \
  --E0s "estimated" \
  --loss="weighted" \
  --stress_weight=0.0 \
  --force_mh_ft_lr=True \
  --default_dtype="float32" \
  --device="cuda" \
  --batch_size=1 \
  --max_num_epochs=20 \
  --multiheads_finetuning=True