#!/bin/bash
#SBATCH --job-name=sal_P10p0000GPa_acquisition_0230_0239
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -n 24
#SBATCH --exclude=bhc0020,bhc0021,bhc0025
#SBATCH --time=120:00:00
#SBATCH -o /gpfs/fs2/scratch/kshao4/mlip/outputsfull/X_semiactivelearning/p10_2gpu_6h/cycles/P10p0000GPa/dft/acquisition/slurm_logs/sal_P10p0000GPa_acquisition_0230_0239_%j.slurmlog.txt
#SBATCH -e /gpfs/fs2/scratch/kshao4/mlip/outputsfull/X_semiactivelearning/p10_2gpu_6h/cycles/P10p0000GPa/dft/acquisition/slurm_logs/sal_P10p0000GPa_acquisition_0230_0239_%j.slurmerr.txt

set -euo pipefail
module load orca/6.1.1
MLIP_DIR="${MLIP_DIR:-/gpfs/fs2/scratch/kshao4/mlip}"
DFT_DIR="$MLIP_DIR/outputsfull/X_semiactivelearning/p10_2gpu_6h/cycles/P10p0000GPa/dft/acquisition"
INPUT_DIR="$MLIP_DIR/codes/X_semiactivelearning/dft_launches/p10_2gpu_6h/P10p0000GPa/acquisition/inputs"
ORCA_COMMAND="${ORCA_COMMAND:-/software/orca/6.1.1/orca}"
cd "$DFT_DIR"
mkdir -p outputs
for stem in cluster_230 cluster_231 cluster_232 cluster_233 cluster_234 cluster_235 cluster_236 cluster_237 cluster_238 cluster_239; do
    output="outputs/${stem}.out"
    if [[ -f "$output" ]] && grep -q "FINAL SINGLE POINT ENERGY" "$output" && grep -q "ORCA TERMINATED NORMALLY" "$output"; then
        echo "Skipping completed $output"
        continue
    fi
    echo "Starting $stem"
    "$ORCA_COMMAND" "$INPUT_DIR/${stem}.inp" > "$output"
    grep -q "FINAL SINGLE POINT ENERGY" "$output"
    grep -q "ORCA TERMINATED NORMALLY" "$output"
    echo "Finished $stem"
done
