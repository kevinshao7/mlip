#!/bin/bash -l
#SBATCH --job-name=mace_polar_debug
#SBATCH --partition=gpudev
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

module purge
module load gcc/10 impi/2021.2
module load anaconda/3/2023.03
module load git/2.50
module load cuda/12.8

source /ptmp/kshao/envs/mlip_clean/bin/activate
export PYTHONNOUSERSITE=1
export PATH=/ptmp/kshao/mlip/packmol:$PATH


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

srun python MHFT.py \
  --name="polar_mhft_1s" \
  --foundation_model="polar-1-s" \
  --train_file="../ab-initio-thermodynamics-of-water/training-set/dataset_1593.xyz" \
  --pt_train_file path/to/replay_dataset.xyz \
  --num_samples_pt 10000 \
  --filter_type_pt combinations \
  --subselect_pt fps \
  --weight_pt 1.0 \
  --atomic_numbers "[1, 7, 8,16]" \
  --energy_key="REF_energy" \
  --forces_key="REF_forces" \
  --stress_key="REF_stress" \
  --compute_forces True \
  --E0s "estimated" \
  --loss="weighted" \
  --stress_weight=0.0 \
  --default_dtype="float32" \
  --device="cuda" \
  --batch_size=1 \
  --max_num_epochs=1 \
  --multiheads_finetuning=True

