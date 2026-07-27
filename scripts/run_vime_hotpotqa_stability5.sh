#!/usr/bin/env bash
# Five-rollout stability gate for the six-GPU ResearchAgent-RL Vime setup.
#
# It deliberately retains one in-flight GRPO group so the run validates
# repeated rollout → reward → update → weight-sync cycles without reviving the
# framework's default 512-group request backlog.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NUM_ROLLOUT="${NUM_ROLLOUT:-5}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export VLLM_SERVER_CONCURRENCY="${VLLM_SERVER_CONCURRENCY:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

exec bash "$SCRIPT_DIR/run_vime_hotpotqa_smoke6.sh"
