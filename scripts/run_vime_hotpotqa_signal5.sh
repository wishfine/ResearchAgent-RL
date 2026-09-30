#!/usr/bin/env bash
# Five-update GRPO signal validation for the six-GPU ResearchAgent-RL setup.
#
# Unlike the two-sample infrastructure smoke, four independent completions per
# prompt provide a meaningful within-group reward comparison.  One group stays
# in flight, preventing request-backlog regressions while keeping memory use
# conservative.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NUM_ROLLOUT="${NUM_ROLLOUT:-5}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export VLLM_SERVER_CONCURRENCY="${VLLM_SERVER_CONCURRENCY:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

exec bash "$SCRIPT_DIR/run_vime_hotpotqa_smoke6.sh"
