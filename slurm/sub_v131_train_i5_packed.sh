#!/usr/bin/env bash
#SBATCH -J v131_i5_pack
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute-g-0
#SBATCH --array=0-3%4
#SBATCH -o outputs/v1_3_1/logs/slurm/%x-%A_%a.log

set -euo pipefail

module load miniconda3/2024.6
module add cuda/12.8
source activate image_trend
ulimit -n 65536

PROJECT_DIR="${IMAGE_TREND_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
OUTPUT_ROOT="${PROJECT_DIR}/outputs/v1_3_1"
export IMAGE_TREND_DATA_DIR="${PROJECT_DIR}/data_v1_3_1"
export IMAGE_TREND_OUTPUT_DIR="${OUTPUT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LOSS="${V131_LOSS:?V131_LOSS must be supplied as bce, huber or huber_ic}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS=(42 43 44 45)
LOG_DIR="${OUTPUT_ROOT}/logs/packed/${LOSS}/task_${TASK_ID}"
mkdir -p "${LOG_DIR}"

run_member() {
    local member_id="$1"
    local mode="$2"
    local fold_id="$((member_id / 4 + 1))"
    local seed_index="$((member_id % 4))"
    local seed="${SEEDS[${seed_index}]}"
    python -u 05_train_cnn2d_v131.py \
        --loss "${LOSS}" --experiments I5R5 \
        --fold-id "${fold_id}" --seed "${seed}" --purge-days 20 \
        --micro-batch-size 0 --max-micro-batch-size 8192 \
        --epochs 25 --warmup-epochs 2 --patience 4 --min-delta 1e-3 \
        --ic-weight 1.0 --huber-beta 1.0 \
        --workers 4 --prefetch-factor 2 --shard-cache-size 4096 \
        --lr 1e-4 --weight-decay 3e-5 --fc-dropout 0.20 \
        --channels-last --log-interval 200 \
        > "${LOG_DIR}/member_${member_id}_${mode}.log" 2>&1
}

START_MEMBER="$((TASK_ID * 5))"
PIDS=()
MEMBERS=()
for OFFSET in 0 1 2 3 4; do
    MEMBER_ID="$((START_MEMBER + OFFSET))"
    run_member "${MEMBER_ID}" packed &
    PIDS+=("$!")
    MEMBERS+=("${MEMBER_ID}")
done

# Do not fail the array immediately if an unusually aligned activation peak
# exhausts a 4080.  Durable members are retained and only missing manifests
# are retried one at a time.
for PID in "${PIDS[@]}"; do
    wait "${PID}" || true
done

for MEMBER_ID in "${MEMBERS[@]}"; do
    FOLD_ID="$((MEMBER_ID / 4 + 1))"
    SEED_INDEX="$((MEMBER_ID % 4))"
    SEED="${SEEDS[${SEED_INDEX}]}"
    MANIFEST="${OUTPUT_ROOT}/manifests/${LOSS}/i5r5/fold$(printf '%02d' "${FOLD_ID}")_seed${SEED}.json"
    if [[ ! -s "${MANIFEST}" ]]; then
        echo "Retrying missing member ${MEMBER_ID} sequentially"
        run_member "${MEMBER_ID}" retry
    fi
done

echo "[$(date --iso-8601=seconds)] packed I5 task complete loss=${LOSS} members=${MEMBERS[*]}"
