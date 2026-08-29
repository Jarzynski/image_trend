#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export IMAGE_TREND_PROJECT_DIR="${ROOT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${ROOT_DIR}/outputs/v1_3_1/logs/slurm"

DEPENDENCY="${V131_INITIAL_DEPENDENCY:-}"
for LOSS in bce huber huber_ic; do
    STAGE_JOBS=()
    if [[ -n "${DEPENDENCY}" ]]; then
        I5_JOB="$(sbatch --parsable --dependency=afterok:${DEPENDENCY} \
            --export=ALL,V131_LOSS=${LOSS} slurm/sub_v131_train_i5_packed.sh)"
    else
        I5_JOB="$(sbatch --parsable --export=ALL,V131_LOSS=${LOSS} \
            slurm/sub_v131_train_i5_packed.sh)"
    fi
    STAGE_JOBS+=("${I5_JOB}")
    echo "${LOSS}/I5R5: train=${I5_JOB} (5 members per GPU)"

    for EXPERIMENT in I20R5 I60R5 I20R20 I60R20; do
        if [[ -n "${DEPENDENCY}" ]]; then
            TRAIN_JOB="$(sbatch --parsable --dependency=afterok:${DEPENDENCY} \
                --export=ALL,V131_LOSS=${LOSS},V131_EXPERIMENTS=${EXPERIMENT} \
                slurm/sub_v131_train.sh)"
        else
            TRAIN_JOB="$(sbatch --parsable \
                --export=ALL,V131_LOSS=${LOSS},V131_EXPERIMENTS=${EXPERIMENT} \
                slurm/sub_v131_train.sh)"
        fi
        STAGE_JOBS+=("${TRAIN_JOB}")
        echo "${LOSS}/${EXPERIMENT}: train=${TRAIN_JOB}"
    done

    STAGE_DEPENDENCY="$(IFS=:; echo "${STAGE_JOBS[*]}")"
    POST_JOB="$(sbatch --parsable --dependency=afterok:${STAGE_DEPENDENCY} \
        --export=ALL,V131_LOSS=${LOSS} slurm/sub_v131_aggregate_backtest.sh)"
    echo "${LOSS}: post=${POST_JOB}"
    DEPENDENCY="${POST_JOB}"
done

echo "Training-only pipeline submitted. Final dependency: ${DEPENDENCY}"
