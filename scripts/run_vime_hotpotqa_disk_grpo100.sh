#!/usr/bin/env bash
# Bounded GRPO training experiment for ResearchAgent-RL on HotpotQA.
#
# This is intentionally a 100-rollout pilot: it validates that the learned
# policy moves under document-grounded rewards before any full 7k-run budget is
# committed.  Four samples per prompt provide a real within-group GRPO signal.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export VLLM_SERVER_CONCURRENCY="${VLLM_SERVER_CONCURRENCY:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"

# The vLLM 0.23 compatibility hook retains the newest two complete HF
# snapshots, pruning only older, already superseded versions.
export UPDATE_WEIGHT_TRANSPORT="disk"
export UPDATE_WEIGHT_DISK_KEEP_FILES="0"
export VIME_DISK_WEIGHT_SYNC_KEEP_LAST="${VIME_DISK_WEIGHT_SYNC_KEEP_LAST:-2}"
# The previous unconstrained pilot collapsed from strict actions into long
# planning prose.  Keep malformed actions well below valid trajectories and
# retain a reference-policy tether.  This launcher is still gated on a
# successful SFT checkpoint evaluation.
export RESEARCH_AGENT_FORMAT_REWARD="${RESEARCH_AGENT_FORMAT_REWARD:-0.20}"
export RESEARCH_AGENT_INVALID_ACTION_PENALTY="${RESEARCH_AGENT_INVALID_ACTION_PENALTY:-0.50}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"

exec bash "$SCRIPT_DIR/run_vime_hotpotqa_smoke6.sh"
