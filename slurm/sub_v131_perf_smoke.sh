#!/usr/bin/env bash
#SBATCH -J v131_perf_smoke
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=96G
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
export IMAGE_TREND_DATA_DIR="${PROJECT_DIR}/data_v1_3_1"
export IMAGE_TREND_OUTPUT_DIR="${PROJECT_DIR}/outputs/v1_3_1_perf_smoke"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "${IMAGE_TREND_OUTPUT_DIR}/logs"
echo "[$(date --iso-8601=seconds)] nofile_soft=$(ulimit -Sn) nofile_hard=$(ulimit -Hn)"

COMMON=(
    --fold-id 1
    --purge-days 20
    --micro-batch-size 0
    --max-micro-batch-size 8192
    --epochs 1
    --warmup-epochs 1
    --patience 1
    --min-delta 1e-3
    --workers 8
    --prefetch-factor 2
    --shard-cache-size 4096
    --log-interval 8
)

if [[ "${V131_PERF_MODE:-full}" == "i60_contiguous" ]]; then
    echo "[$(date --iso-8601=seconds)] RTX 4080 I60R20 contiguous comparison"
    python -u 05_train_cnn2d_v131.py \
        --loss huber_ic --experiments I60R20 --smoke-dates 8 --seed 143 \
        "${COMMON[@]}"
    echo "[$(date --iso-8601=seconds)] contiguous comparison complete"
    exit 0
fi

if [[ "${V131_PERF_MODE:-full}" == "i60_batch_probe" ]]; then
    PROBE_BATCH="${V131_PROBE_BATCH:?V131_PROBE_BATCH is required for i60_batch_probe}"
    PROBE_SEED="${V131_PROBE_SEED:-$((200 + PROBE_BATCH))}"
    PROBE_DATES="${V131_PROBE_DATES:-8}"
    echo "[$(date --iso-8601=seconds)] RTX 4080 I60R20 Huber+IC batch=${PROBE_BATCH} probe"
    python -u 05_train_cnn2d_v131.py \
        --loss huber_ic --experiments I60R20 --smoke-dates "${PROBE_DATES}" \
        --seed "${PROBE_SEED}" --micro-batch-size "${PROBE_BATCH}" \
        --fold-id 1 --purge-days 20 --max-micro-batch-size 8192 \
        --epochs 1 --warmup-epochs 1 --patience 1 --min-delta 1e-3 \
        --workers 8 --prefetch-factor 2 --shard-cache-size 4096 \
        --log-interval 8 --channels-last
    echo "[$(date --iso-8601=seconds)] batch=${PROBE_BATCH} probe complete"
    exit 0
fi

echo "[$(date --iso-8601=seconds)] RTX 4080 I5R5 BCE throughput smoke"
python -u 05_train_cnn2d_v131.py \
    --loss bce --experiments I5R5 --smoke-dates 48 --seed 142 \
    --channels-last "${COMMON[@]}"

echo "[$(date --iso-8601=seconds)] RTX 4080 I60R20 Huber+IC memory smoke"
python -u 05_train_cnn2d_v131.py \
    --loss huber_ic --experiments I60R20 --smoke-dates 8 --seed 142 \
    --channels-last "${COMMON[@]}"

echo "[$(date --iso-8601=seconds)] performance smoke complete"
