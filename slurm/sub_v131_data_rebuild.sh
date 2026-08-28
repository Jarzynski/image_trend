#!/usr/bin/env bash
#SBATCH -J v131_data
#SBATCH -p single
#SBATCH -N 1
#SBATCH -c 24
# The single partition nodes expose about 32GB each; the build scripts stream
# partitions and keep their worker count bounded to fit that allocation.
#SBATCH --mem=30G
#SBATCH -o outputs/v1_3_1/logs/slurm/%x-%j.log

set -euo pipefail

module load miniconda3/2024.6
source activate image_trend

PROJECT_DIR="${IMAGE_TREND_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"
DATA_ROOT="${PROJECT_DIR}/data_v1_3_1"
OUTPUT_ROOT="${PROJECT_DIR}/outputs/v1_3_1"
export IMAGE_TREND_DATA_DIR="${DATA_ROOT}"
export IMAGE_TREND_OUTPUT_DIR="${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CPU_WORKERS="${SLURM_CPUS_PER_TASK:-24}"
mkdir -p "${OUTPUT_ROOT}/logs/slurm" "${OUTPUT_ROOT}/tables/qa"
echo "[$(date --iso-8601=seconds)] project=${PROJECT_DIR} data=${DATA_ROOT} output=${OUTPUT_ROOT} workers=${CPU_WORKERS}"
echo "[$(date --iso-8601=seconds)] rebuilding panel 2009-2024"
python -u 01_build_panel.py --workers "${CPU_WORKERS}"
echo "[$(date --iso-8601=seconds)] rebuilding features 2009-2024"
python -u 02_make_labels_and_baselines.py --workers "${CPU_WORKERS}"
echo "[$(date --iso-8601=seconds)] rebuilding window images"
python -u 03_make_images.py --workers "${CPU_WORKERS}"
echo "[$(date --iso-8601=seconds)] running data QA"
python -u 08_qa_v131.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --old-data-dir "${PROJECT_DIR}/data" \
    --output "${OUTPUT_ROOT}/tables/qa/data_qa_v131.json"
echo "[$(date --iso-8601=seconds)] data rebuild and QA complete"
