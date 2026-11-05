#!/bin/bash -l
#SBATCH --job-name=X_P10_DFT_195
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/viper/ptmp1/kshao/mlip/outputsfull/X_semiactivelearning/P10GPa/cpu_dft_results/p10_2000_viper_20261105T164509178672Z/slurm/group_195_%j.out
#SBATCH --error=/viper/ptmp1/kshao/mlip/outputsfull/X_semiactivelearning/P10GPa/cpu_dft_results/p10_2000_viper_20261105T164509178672Z/slurm/group_195_%j.err
set -euo pipefail
module purge
module load orca/6.1.1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME=/viper/ptmp1/kshao/mlip/outputsfull/.cache
cd /viper/ptmp1/kshao/mlip
"${PYTHON_COMMAND:-/viper/ptmp1/kshao/mlip/.venv/bin/python}" /viper/ptmp1/kshao/mlip/codes/X_semiactivelearning/dft/viper_cpu.py --package /viper/ptmp1/kshao/mlip/codes/X_semiactivelearning/dft/batches/p10_2000_viper_20261105T164509178672Z/package.json run-group --index 195
