#!/usr/bin/env bash
#SBATCH -J v131_i5_pack
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH --mem=160G
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --exclude=compute-g-0
#SBATCH -o outputs/v1_3_1/logs/slurm/%x-%j.log

set -euo pipefail

module load miniconda3/2024.6
module add cuda/12.8
source activate image_trend
ulimit -n 65536

PROJECT_DIR="${IMAGE_TREND_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
LANES="${V131_PACK_LANES:?V131_PACK_LANES must be 2, 3, 4 or 5}"
if (( LANES < 2 || LANES > 5 )); then
    echo "Invalid V131_PACK_LANES=${LANES}" >&2
    exit 2
fi

export IMAGE_TREND_DATA_DIR="${PROJECT_DIR}/data_v1_3_1"
export IMAGE_TREND_OUTPUT_DIR="${PROJECT_DIR}/outputs/v1_3_1_pack_probe/lanes_${LANES}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "${IMAGE_TREND_OUTPUT_DIR}/process_logs"

START_SECONDS="$(date +%s)"
PIDS=()
for ((LANE=0; LANE<LANES; LANE++)); do
    SEED="$((3000 + LANES * 10 + LANE))"
    python -u 05_train_cnn2d_v131.py \
        --loss bce --experiments I5R5 --fold-id 5 --seed "${SEED}" \
        --smoke-dates 64 --smoke-date-position tail \
        --micro-batch-size 0 --max-micro-batch-size 8192 \
        --epochs 1 --warmup-epochs 1 --patience 1 --min-delta 1e-3 \
        --workers 4 --prefetch-factor 2 --shard-cache-size 4096 \
        --channels-last --log-interval 16 \
        > "${IMAGE_TREND_OUTPUT_DIR}/process_logs/lane_${LANE}.log" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAILED=1
    fi
done
ELAPSED="$(( $(date +%s) - START_SECONDS ))"
echo "lanes=${LANES} elapsed_seconds=${ELAPSED} failed=${FAILED}"
grep -h -E 'Start bce|Epoch 01|CUDA out|OutOfMemory|Training complete' \
    "${IMAGE_TREND_OUTPUT_DIR}"/process_logs/lane_*.log || true
exit "${FAILED}"
