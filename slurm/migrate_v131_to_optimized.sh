#!/usr/bin/env bash

# Wait until the two expensive legacy members are fully durable, then replace
# the old 4090-only dependency chain with the optimized 4080/4090 chain.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STATE_DIR="${ROOT_DIR}/outputs/v1_3_1/logs/migration"
LOCK_DIR="${STATE_DIR}/migration.lock"
DONE_FILE="${STATE_DIR}/optimized_pipeline_submitted.txt"
OLD_JOB_IDS="${V131_OLD_JOB_IDS:-84645 84646 84647 84648 84649 84650}"
POLL_SECONDS="${V131_MIGRATION_POLL_SECONDS:-30}"
mkdir -p "${STATE_DIR}"

if [[ -s "${DONE_FILE}" ]]; then
    cat "${DONE_FILE}"
    exit 0
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Another migration watcher owns ${LOCK_DIR}" >&2
    exit 3
fi

READY_FILES=(
    outputs/v1_3_1/manifests/bce/i5r5/fold01_seed42.json
    outputs/v1_3_1/models/bce/i5r5/fold01_seed42.pt
    outputs/v1_3_1/predictions/members/bce/i5r5/fold01_seed42.parquet
    outputs/v1_3_1/manifests/bce/i5r5/fold01_seed43.json
    outputs/v1_3_1/models/bce/i5r5/fold01_seed43.pt
    outputs/v1_3_1/predictions/members/bce/i5r5/fold01_seed43.parquet
)

echo "[$(date --iso-8601=seconds)] waiting for legacy fold01 seeds 42 and 43"
while true; do
    READY=1
    for FILE in "${READY_FILES[@]}"; do
        if [[ ! -s "${FILE}" ]]; then
            READY=0
            break
        fi
    done
    if [[ "${READY}" -eq 1 ]]; then
        break
    fi
    # The array parent must remain visible while its required members are not
    # durable.  If it vanishes, fail closed instead of launching a duplicate
    # chain with an ambiguous checkpoint state.
    if ! squeue -h -j 84645 | grep -q .; then
        echo "[$(date --iso-8601=seconds)] old array vanished before required files were durable" >&2
        exit 4
    fi
    sleep "${POLL_SECONDS}"
done

echo "[$(date --iso-8601=seconds)] both legacy members durable; cancelling exact old chain"
# shellcheck disable=SC2086
scancel ${OLD_JOB_IDS}
for _ in $(seq 1 60); do
    # shellcheck disable=SC2086
    if ! squeue -h -j "$(tr ' ' ',' <<< "${OLD_JOB_IDS}")" | grep -q .; then
        break
    fi
    sleep 2
done

SUBMIT_LOG="${STATE_DIR}/optimized_submit_$(date +%Y%m%d_%H%M%S).log"
bash slurm/submit_v131_training_only.sh | tee "${SUBMIT_LOG}"
{
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "preserved_members=fold01_seed42,fold01_seed43"
    echo "cancelled_jobs=${OLD_JOB_IDS}"
    echo "submit_log=${SUBMIT_LOG}"
} > "${DONE_FILE}.tmp"
mv "${DONE_FILE}.tmp" "${DONE_FILE}"
cat "${DONE_FILE}"
