#!/usr/bin/env bash

# Detached, idempotent launcher for the v1.3.1 Slurm dependency chain.
#
# The first invocation daemonizes itself and returns immediately.  The child
# validates the already-submitted data/smoke jobs, submits the strict
# afterok pipeline, and records an atomic completion marker.  A start marker
# is deliberately retained on failure so a partial submission cannot be
# accidentally duplicated by a later retry.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_JOB="${V131_DATA_JOB_ID:-${1:-}}"
SMOKE_JOB="${V131_SMOKE_JOB_ID:-${2:-}}"
if [[ -z "${DATA_JOB}" || -z "${SMOKE_JOB}" ]]; then
    echo "Usage: V131_DATA_JOB_ID=<data-job> V131_SMOKE_JOB_ID=<smoke-job> $0"
    echo "   or: $0 <data-job> <smoke-job>"
    exit 2
fi
if [[ ! "${DATA_JOB}" =~ ^[0-9]+$ || ! "${SMOKE_JOB}" =~ ^[0-9]+$ ]]; then
    echo "Job IDs must be numeric (data=${DATA_JOB}, smoke=${SMOKE_JOB})" >&2
    exit 2
fi

STATE_DIR="${ROOT_DIR}/outputs/v1_3_1/logs/background"
mkdir -p "${STATE_DIR}"
START_MARKER="${STATE_DIR}/pipeline_started.txt"
SUBMITTED_MARKER="${STATE_DIR}/pipeline_submitted.txt"
LOCK_DIR="${STATE_DIR}/pipeline_submit.lock"
PID_FILE="${STATE_DIR}/launcher.pid"

if [[ -f "${SUBMITTED_MARKER}" ]]; then
    echo "v1.3.1 pipeline already submitted: ${SUBMITTED_MARKER}"
    cat "${SUBMITTED_MARKER}"
    exit 0
fi
if [[ -f "${START_MARKER}" ]]; then
    echo "v1.3.1 pipeline launch is already recorded; refusing a duplicate submission: ${START_MARKER}"
    cat "${START_MARKER}"
    exit 0
fi

# The foreground parent owns the atomic lock while it starts the detached
# child.  Concurrent SSH invocations therefore cannot start two children.
if [[ "${V131_BACKGROUND_CHILD:-0}" != "1" ]]; then
    if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
        echo "Another v1.3.1 launcher is already starting (${LOCK_DIR})."
        exit 0
    fi
    stamp="$(date +%Y%m%d_%H%M%S)"
    launch_log="${STATE_DIR}/launcher_${stamp}.log"
    # Starting an asynchronous process returns before the child has finished;
    # do not negate the command here, otherwise Bash reports a false failure
    # even when the detached child submits the Slurm jobs successfully.
    nohup env \
        V131_BACKGROUND_CHILD=1 \
        V131_LOCK_HELD=1 \
        V131_DATA_JOB_ID="${DATA_JOB}" \
        V131_SMOKE_JOB_ID="${SMOKE_JOB}" \
        bash "${BASH_SOURCE[0]}" "${DATA_JOB}" "${SMOKE_JOB}" \
        >"${launch_log}" 2>&1 < /dev/null &
    child_pid="$!"
    pid_tmp="${PID_FILE}.tmp.$$"
    {
        echo "pid=${child_pid}"
        echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
        echo "data_job=${DATA_JOB}"
        echo "smoke_job=${SMOKE_JOB}"
        echo "log=${launch_log}"
    } > "${pid_tmp}"
    mv -f "${pid_tmp}" "${PID_FILE}"
    echo "Detached v1.3.1 launcher started (pid=${child_pid})."
    echo "Log: ${launch_log}"
    echo "State: ${STATE_DIR}"
    exit 0
fi

if [[ "${V131_LOCK_HELD:-0}" != "1" ]]; then
    if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
        echo "Another v1.3.1 launcher is already running (${LOCK_DIR})."
        exit 0
    fi
fi

cleanup() {
    local status=$?
    if (( status == 0 )); then
        # The marker is the durable idempotency guard after successful submit.
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    else
        echo "Background pipeline launcher failed with status ${status}." >&2
        echo "The start marker, if present, is retained to prevent duplicate jobs." >&2
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available; load the Slurm environment before launching." >&2
    exit 1
fi
if ! command -v scontrol >/dev/null 2>&1; then
    echo "scontrol is not available; refusing to submit without job validation." >&2
    exit 1
fi
if ! scontrol show job "${DATA_JOB}" >/dev/null 2>&1; then
    echo "Cannot validate data job ${DATA_JOB} with scontrol." >&2
    exit 1
fi
if ! scontrol show job "${SMOKE_JOB}" >/dev/null 2>&1; then
    echo "Cannot validate smoke job ${SMOKE_JOB} with scontrol." >&2
    exit 1
fi

start_tmp="${START_MARKER}.tmp.$$"
{
    echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "launcher_pid=$$"
    echo "data_job=${DATA_JOB}"
    echo "smoke_job=${SMOKE_JOB}"
    echo "root=${ROOT_DIR}"
} > "${start_tmp}"
mv -f "${start_tmp}" "${START_MARKER}"

echo "Validated data job ${DATA_JOB} and smoke job ${SMOKE_JOB}."
echo "Submitting v1.3.1 data-gated, smoke-gated loss pipeline."
V131_DATA_JOB_ID="${DATA_JOB}" \
V131_SMOKE_JOB_ID="${SMOKE_JOB}" \
bash "${ROOT_DIR}/slurm/submit_v131_pipeline.sh"

submitted_tmp="${SUBMITTED_MARKER}.tmp.$$"
{
    echo "submitted_at=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "data_job=${DATA_JOB}"
    echo "smoke_job=${SMOKE_JOB}"
    echo "launcher_pid=$$"
    echo "log=${STATE_DIR}"
} > "${submitted_tmp}"
mv -f "${submitted_tmp}" "${SUBMITTED_MARKER}"
echo "Pipeline submission recorded at ${SUBMITTED_MARKER}."
