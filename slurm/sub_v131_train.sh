#!/usr/bin/env bash
#SBATCH -J v131_train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --exclude=compute-g-0
#SBATCH --array=0-19%8
#SBATCH -o outputs/v1_3_1/logs/slurm/%x-%A_%a.log

set -euo pipefail

module load miniconda3/2024.6
module add cuda/12.8
source activate image_trend

PROJECT_DIR="${IMAGE_TREND_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
DATA_ROOT="${PROJECT_DIR}/data_v1_3_1"
OUTPUT_ROOT="${PROJECT_DIR}/outputs/v1_3_1"
export IMAGE_TREND_DATA_DIR="${DATA_ROOT}"
export IMAGE_TREND_OUTPUT_DIR="${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOSS="${V131_LOSS:?V131_LOSS must be supplied as bce, huber or huber_ic}"
SEEDS=(42 43 44 45)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FOLD_ID="$((TASK_ID / 4 + 1))"
SEED_INDEX="$((TASK_ID % 4))"
SEED="${SEEDS[${SEED_INDEX}]}"

mkdir -p "${OUTPUT_ROOT}/logs/slurm"
echo "[$(date --iso-8601=seconds)] loss=${LOSS} fold=${FOLD_ID} seed=${SEED} host=$(hostname)"
python -u 05_train_cnn2d_v131.py \
    --loss "${LOSS}" \
    --fold-id "${FOLD_ID}" \
    --seed "${SEED}" \
    --purge-days 20 \
    --micro-batch-size 256 \
    --epochs 25 \
    --warmup-epochs 2 \
    --patience 4 \
    --min-delta 1e-3 \
    --ic-weight 1.0 \
    --huber-beta 1.0 \
    --workers 2 \
    --prefetch-factor 2 \
    --shard-cache-size 32 \
    --lr 1e-4 \
    --weight-decay 3e-5 \
    --fc-dropout 0.20 \
    --log-interval 200
echo "[$(date --iso-8601=seconds)] training task complete"
