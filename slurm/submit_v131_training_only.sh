#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export IMAGE_TREND_PROJECT_DIR="${ROOT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${ROOT_DIR}/outputs/v1_3_1/logs/slurm"

DEPENDENCY=""
for LOSS in bce huber huber_ic; do
    if [[ -n "${DEPENDENCY}" ]]; then
        TRAIN_JOB="$(sbatch --parsable --dependency=afterok:${DEPENDENCY} \
            --export=ALL,V131_LOSS=${LOSS} slurm/sub_v131_train.sh)"
    else
        TRAIN_JOB="$(sbatch --parsable --export=ALL,V131_LOSS=${LOSS} \
            slurm/sub_v131_train.sh)"
    fi
    POST_JOB="$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
        --export=ALL,V131_LOSS=${LOSS} slurm/sub_v131_aggregate_backtest.sh)"
    echo "${LOSS}: train=${TRAIN_JOB} post=${POST_JOB}"
    DEPENDENCY="${POST_JOB}"
done

echo "Training-only pipeline submitted. Final dependency: ${DEPENDENCY}"
