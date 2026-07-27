#!/usr/bin/env bash
# Five-rollout regression test for the vLLM 0.23 disk-sync retention policy.
#
# The compatibility hook keeps the newest two checkpoint versions and prunes
# only fully superseded versions.  Keep-files is intentionally 0 here to prove
# that the hook, rather than unbounded checkpoint retention, prevents the
# shard-deletion race.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NUM_ROLLOUT="${NUM_ROLLOUT:-5}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export VLLM_SERVER_CONCURRENCY="${VLLM_SERVER_CONCURRENCY:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export UPDATE_WEIGHT_TRANSPORT="disk"
export UPDATE_WEIGHT_DISK_KEEP_FILES="0"
export VIME_DISK_WEIGHT_SYNC_KEEP_LAST="${VIME_DISK_WEIGHT_SYNC_KEEP_LAST:-2}"

exec bash "$SCRIPT_DIR/run_vime_hotpotqa_smoke6.sh"
