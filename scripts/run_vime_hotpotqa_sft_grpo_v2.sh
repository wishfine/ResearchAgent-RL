#!/usr/bin/env bash
# GRPO v2: continue actor and KL reference from the validated SFT-874 policy.
#
# GPU topology is owned by run_vime_hotpotqa_smoke6.sh:
# GPUs 2-5 actor (TP=2, DP=2); GPUs 6-7 rollout vLLM (TP=2).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-/data/$USER/research-agent-rl-data}"
SFT_RUN="${SFT_RUN:-$BASE/outputs/vime_hotpotqa_sft7k_20260721_162728}"
SFT_CHECKPOINTS="${SFT_CHECKPOINTS:-$SFT_RUN/checkpoints}"

[[ -f "$SFT_CHECKPOINTS/latest_checkpointed_iteration.txt" ]] || {
  echo "[ERROR] SFT checkpoint index not found: $SFT_CHECKPOINTS/latest_checkpointed_iteration.txt" >&2
  exit 1
}

# Both model roles start at SFT-874: the actor is trainable and ref-load is
# frozen, so KL penalizes movement away from the validated SFT policy.
export REF_CHECKPOINT="$SFT_CHECKPOINTS"
export ACTOR_LOAD="$SFT_CHECKPOINTS"
export CKPT_STEP="${CKPT_STEP:-874}"

# One rollout group has one prompt and four policy samples; this supplies an
# actual within-group GRPO comparison while retaining the validated topology.
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export VLLM_SERVER_CONCURRENCY="${VLLM_SERVER_CONCURRENCY:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"

# The first base-model GRPO run used KL=0 and lost strict action formatting.
# Keep valid formatted trajectories distinguishable from invalid ones and
# tether the actor to the SFT reference policy.
export RESEARCH_AGENT_FORMAT_REWARD="${RESEARCH_AGENT_FORMAT_REWARD:-0.20}"
export RESEARCH_AGENT_INVALID_ACTION_PENALTY="${RESEARCH_AGENT_INVALID_ACTION_PENALTY:-0.50}"
export RESEARCH_AGENT_STEP_PENALTY="${RESEARCH_AGENT_STEP_PENALTY:-0.01}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"

# The installed vLLM 0.23 only has a validated full checkpoint reload path.
export UPDATE_WEIGHT_TRANSPORT="disk"
export UPDATE_WEIGHT_DISK_KEEP_FILES="0"
export VIME_DISK_WEIGHT_SYNC_KEEP_LAST="${VIME_DISK_WEIGHT_SYNC_KEEP_LAST:-2}"

exec bash "$SCRIPT_DIR/run_vime_hotpotqa_smoke6.sh"
