#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export IMAGE_TREND_PROJECT_DIR="${ROOT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${ROOT_DIR}/outputs/v1_3_1/logs/slurm"

echo "Project dir: ${ROOT_DIR}"
if [[ -n "${V131_DATA_JOB_ID:-}" ]]; then
    DATA_JOB="${V131_DATA_JOB_ID}"
    echo "Reusing v1.3.1 data rebuild job: ${DATA_JOB}"
else
    echo "Submitting v1.3.1 data rebuild"
    DATA_JOB="$(sbatch --parsable slurm/sub_v131_data_rebuild.sh)"
    echo "  data job: ${DATA_JOB}"
fi

DEPENDENCY="${DATA_JOB}"
if [[ -n "${V131_SMOKE_JOB_ID:-}" ]]; then
    # Gate the first loss stage on both the completed data rebuild and the
    # smoke test.  Later loss stages remain chained to the preceding post job.
    if [[ ! "${V131_SMOKE_JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "Invalid V131_SMOKE_JOB_ID: ${V131_SMOKE_JOB_ID}" >&2
        exit 2
    fi
    DEPENDENCY="${DATA_JOB}:${V131_SMOKE_JOB_ID}"
    echo "  smoke dependency: ${V131_SMOKE_JOB_ID}"
fi
echo "Submitting packed/experiment-split training chain after: ${DEPENDENCY}"
V131_INITIAL_DEPENDENCY="${DEPENDENCY}" bash slurm/submit_v131_training_only.sh
