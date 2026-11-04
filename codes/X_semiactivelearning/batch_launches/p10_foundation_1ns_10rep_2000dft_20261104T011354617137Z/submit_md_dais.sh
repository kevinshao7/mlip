#!/bin/bash
set -euo pipefail
export MLIP_ROOT=${MLIP_ROOT:-/dais/fs/scratch/kshao/mlip}
source "${MLIP_ENV:-/dais/fs/scratch/kshao/mlipenv}/bin/activate"
cd "$MLIP_ROOT"
python codes/X_semiactivelearning/p10_batch.py --plan "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_1ns_10rep_2000dft_20261104T011354617137Z/plan.json" check-md
mkdir -p "$MLIP_ROOT/outputsfull/X_semiactivelearning/p10_foundation_1ns_10rep_2000dft_20261104T011354617137Z/slurm"
for replica in $(seq 0 9); do
    printf -v replica_tag '%02d' "$replica"
    job=$(sbatch --parsable "$MLIP_ROOT/codes/X_semiactivelearning/batch_launches/p10_foundation_1ns_10rep_2000dft_20261104T011354617137Z/md_replica_${replica_tag}_dais.slurm")
    job=${job%%;*}
    echo "Replica $replica_tag submitted as independent job $job"
done
