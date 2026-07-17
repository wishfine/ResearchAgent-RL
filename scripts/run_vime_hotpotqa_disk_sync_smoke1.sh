#!/usr/bin/env bash
# Exercise Vime's full-checkpoint disk sync for Qwen3.5.
# This avoids the unverified incremental NCCL/layerwise reload path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-9999}"
export VLLM_SERVER_CONCURRENCY="${VLLM_SERVER_CONCURRENCY:-1}"
export UPDATE_WEIGHT_TRANSPORT="disk"
export UPDATE_WEIGHT_DISK_KEEP_FILES="${UPDATE_WEIGHT_DISK_KEEP_FILES:-1}"

exec bash "$SCRIPT_DIR/run_vime_hotpotqa_smoke6.sh"
