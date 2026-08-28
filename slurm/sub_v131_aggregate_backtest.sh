#!/usr/bin/env bash
#SBATCH -J v131_post
#SBATCH -p single
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=30G
#SBATCH -o outputs/v1_3_1/logs/slurm/%x-%j.log

set -euo pipefail

module load miniconda3/2024.6
source activate image_trend

PROJECT_DIR="${IMAGE_TREND_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
DATA_ROOT="${PROJECT_DIR}/data_v1_3_1"
OUTPUT_ROOT="${PROJECT_DIR}/outputs/v1_3_1"
PRED_ROOT="${OUTPUT_ROOT}/predictions"
export IMAGE_TREND_DATA_DIR="${DATA_ROOT}"
export IMAGE_TREND_OUTPUT_DIR="${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

LOSS="${V131_LOSS:?V131_LOSS must be supplied as bce, huber or huber_ic}"
DECILE_TABLE_DIR="${OUTPUT_ROOT}/tables/decile/${LOSS}"
NONOVERLAP_TABLE_DIR="${OUTPUT_ROOT}/tables/nonoverlap/${LOSS}"
mkdir -p "${OUTPUT_ROOT}/tables/ensemble" "${DECILE_TABLE_DIR}" "${NONOVERLAP_TABLE_DIR}" "${OUTPUT_ROOT}/logs/slurm"

echo "[$(date --iso-8601=seconds)] aggregating ${LOSS}"
python -u aggregate_cnn_v131.py --loss "${LOSS}" --expected-members 20
echo "[$(date --iso-8601=seconds)] running decile backtests for ${LOSS}"
python -u 06_backtest_decile.py \
    --pred-dir "${PRED_ROOT}" \
    --table-dir "${DECILE_TABLE_DIR}" \
    --pred-pattern "pred_*_jiang_cnn2d_v131_${LOSS}_k5s4.parquet"
echo "[$(date --iso-8601=seconds)] running non-overlap backtests for ${LOSS}"
python -u 07_backtest_nonoverlap_portfolio.py \
    --pred-dir "${PRED_ROOT}" \
    --table-dir "${NONOVERLAP_TABLE_DIR}" \
    --pred-pattern "pred_*_jiang_cnn2d_v131_${LOSS}_k5s4.parquet"
echo "[$(date --iso-8601=seconds)] aggregate and backtests complete for ${LOSS}"
