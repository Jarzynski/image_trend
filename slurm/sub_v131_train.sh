#!/usr/bin/env bash
#SBATCH -J v131_train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-g-0
#SBATCH --array=0-19%16
#SBATCH -o outputs/v1_3_1/logs/slurm/%x-%A_%a.log

set -euo pipefail

module load miniconda3/2024.6
module add cuda/12.8
source activate image_trend
ulimit -n 65536

PROJECT_DIR="${IMAGE_TREND_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
DATA_ROOT="${PROJECT_DIR}/data_v1_3_1"
OUTPUT_ROOT="${PROJECT_DIR}/outputs/v1_3_1"
export IMAGE_TREND_DATA_DIR="${DATA_ROOT}"
export IMAGE_TREND_OUTPUT_DIR="${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LOSS="${V131_LOSS:?V131_LOSS must be supplied as bce, huber or huber_ic}"
EXPERIMENTS="${V131_EXPERIMENTS:-}"
SEEDS=(42 43 44 45)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FOLD_ID="$((TASK_ID / 4 + 1))"
SEED_INDEX="$((TASK_ID % 4))"
SEED="${SEEDS[${SEED_INDEX}]}"

mkdir -p "${OUTPUT_ROOT}/logs/slurm"
echo "[$(date --iso-8601=seconds)] nofile_soft=$(ulimit -Sn) nofile_hard=$(ulimit -Hn)"
echo "[$(date --iso-8601=seconds)] loss=${LOSS} fold=${FOLD_ID} seed=${SEED} host=$(hostname)"
EXPERIMENT_ARGS=()
if [[ -n "${EXPERIMENTS}" ]]; then
    EXPERIMENT_ARGS=(--experiments "${EXPERIMENTS}")
fi
python -u 05_train_cnn2d_v131.py \
    --loss "${LOSS}" \
    --fold-id "${FOLD_ID}" \
    --seed "${SEED}" \
    "${EXPERIMENT_ARGS[@]}" \
    --purge-days 20 \
    --micro-batch-size 0 \
    --max-micro-batch-size 8192 \
    --epochs 25 \
    --warmup-epochs 2 \
    --patience 4 \
    --min-delta 1e-3 \
    --ic-weight 1.0 \
    --huber-beta 1.0 \
    --workers 8 \
    --prefetch-factor 2 \
    --shard-cache-size 4096 \
    --lr 1e-4 \
    --weight-decay 3e-5 \
    --fc-dropout 0.20 \
    --channels-last \
    --log-interval 200
echo "[$(date --iso-8601=seconds)] training task complete"
