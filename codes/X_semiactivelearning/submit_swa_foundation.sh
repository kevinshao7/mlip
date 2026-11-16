#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case ${1:-} in
    --dry-run|'') ;;
    *) echo 'Usage: bash submit_swa_foundation.sh [--dry-run]' >&2; exit 2 ;;
esac
for model in s m l; do
    command=(sbatch --job-name="X_P10_foundSWA_1${model}" finetune_swa_foundation_dais.slurm --model "$model")
    if [[ ${1:-} == --dry-run ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done
