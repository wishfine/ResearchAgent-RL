#!/usr/bin/env bash
# Native Vime/Megatron SFT for strict HotpotQA tool-use trajectories.
#
# This intentionally uses no vLLM/rollout GPUs.  With GPU_IDS=2,3,4,5 it
# forms TP=2, DP=2 and leaves GPUs 0,1,6,7 untouched.

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-/data/$USER/research-agent-rl-data}"
VIME_ROOT="${VIME_ROOT:-/data/$USER/vime-src/vime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/data/$USER/vime-src/Megatron-LM}"
TRAIN_ENV="${TRAIN_ENV:-/data/$USER/conda_envs/vime-train-cu129}"
SITE="${SITE:-$TRAIN_ENV/lib/python3.11/site-packages}"
CUDA_HOME="${CUDA_HOME:-/data/$USER/cuda-toolkit-12.9/usr/local/cuda-12.9}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/home/$USER/models/models/Qwen--Qwen3.5-9B/snapshots/master}"
REF_CHECKPOINT="${REF_CHECKPOINT:-$BASE/checkpoints/Qwen3.5-9B_torch_dist_v4_noapex}"
SFT_PROMPT_DATA="${SFT_PROMPT_DATA:-$BASE/sft_data/hotpotqa_train_7k_strict_actions.jsonl}"
RUN_DIR="${RUN_DIR:-$BASE/outputs/vime_hotpotqa_sft_$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-2,3,4,5}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
SFT_NUM_EPOCH="${SFT_NUM_EPOCH:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

fail() { echo "[ERROR] $*" >&2; exit 1; }
IFS=',' read -r -a GPU_LIST <<<"$GPU_IDS"
[[ ${#GPU_LIST[@]} -eq "$ACTOR_GPUS" ]] || fail "GPU_IDS must contain $ACTOR_GPUS GPU IDs"
[[ "$ACTOR_GPUS" == "4" ]] || fail "This validated launcher requires ACTOR_GPUS=4"
[[ -x "$TRAIN_ENV/bin/python" ]] || fail "Training Python not found: $TRAIN_ENV/bin/python"
[[ -d "$VIME_ROOT" ]] || fail "Vime source not found: $VIME_ROOT"
[[ -d "$MEGATRON_ROOT" ]] || fail "Megatron source not found: $MEGATRON_ROOT"
[[ -d "$HF_CHECKPOINT" ]] || fail "HF checkpoint not found: $HF_CHECKPOINT"
[[ -d "$REF_CHECKPOINT" ]] || fail "Converted checkpoint not found: $REF_CHECKPOINT"
[[ -f "$SFT_PROMPT_DATA" ]] || fail "SFT prompt data not found: $SFT_PROMPT_DATA"
[[ "$SFT_NUM_EPOCH" =~ ^[1-9][0-9]*$ ]] || fail "SFT_NUM_EPOCH must be positive"
[[ "$ROLLOUT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "ROLLOUT_BATCH_SIZE must be positive"
[[ "$GLOBAL_BATCH_SIZE" == "$ROLLOUT_BATCH_SIZE" ]] || fail "SFT requires GLOBAL_BATCH_SIZE == ROLLOUT_BATCH_SIZE"

"$TRAIN_ENV/bin/python" - "$SFT_PROMPT_DATA" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    first = json.loads(next(line for line in handle if line.strip()))
messages = first.get("messages")
if not isinstance(messages, list) or not messages:
    raise SystemExit("SFT record must contain non-empty messages")
roles = [message.get("role") for message in messages]
if any(role not in {"system", "user", "assistant"} for role in roles):
    raise SystemExit(f"SFT records must use standard chat roles, got {sorted(set(roles))}")
assistant = [message["content"] for message in messages if message.get("role") == "assistant"]
if len(assistant) != 4 or not all(text.startswith("<action>{") and text.endswith("}</action>") for text in assistant):
    raise SystemExit("SFT record does not contain four strict compact assistant actions")
print("SFT prompt format: standard messages with four strict actions OK")
PY

for gpu in "${GPU_LIST[@]}"; do
  processes="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')"
  [[ -z "$processes" ]] || fail "GPU $gpu is busy (PID(s): $processes)"
done

NVIDIA_LIBRARY_PATH="$(find "$SITE/nvidia" -type d -name lib -print | paste -sd: -)"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_HOME CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT:$MEGATRON_ROOT:$VIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/$USER/ray}"
mkdir -p "$RUN_DIR" "$RAY_TMPDIR"

"$TRAIN_ENV/bin/python" - <<'PY'
import ray
import transformer_engine.pytorch
import vime
from vime.rollout.sft_rollout import generate_rollout

print("Vime native SFT preflight: OK")
print("ray:", ray.__version__)
print("vime:", vime.__file__)
PY

if "$TRAIN_ENV/bin/ray" status --address="127.0.0.1:$RAY_PORT" >/dev/null 2>&1; then
  fail "A Ray cluster is already listening on port $RAY_PORT; stop only the intended cluster first"
fi

ray_started=0
cleanup_ray() {
  local status=$?
  if [[ "$ray_started" -eq 1 ]]; then
    RAY_TMPDIR="$RAY_TMPDIR" "$TRAIN_ENV/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  return "$status"
}
trap cleanup_ray EXIT

source "$VIME_ROOT/scripts/models/qwen3.5-9B.sh"
"$TRAIN_ENV/bin/ray" start --head --node-ip-address 127.0.0.1 --port "$RAY_PORT" \
  --dashboard-port "$RAY_DASHBOARD_PORT" --num-gpus "$ACTOR_GPUS" --disable-usage-stats
ray_started=1

for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$RAY_DASHBOARD_PORT/api/version" >/dev/null && break
  sleep 1
done

RUNTIME_ENV_JSON="$({ PYTHONPATH="" "$TRAIN_ENV/bin/python" - <<PY
import json
import os
print(json.dumps({"env_vars": {
    "PYTHONPATH": "${PYTHONPATH}",
    "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
    "CUDA_HOME": os.environ["CUDA_HOME"],
    "CUDA_PATH": os.environ["CUDA_PATH"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
}}))
PY
} )"

CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_CHECKPOINT"
  --save "$RUN_DIR/checkpoints"
  --save-interval "$SAVE_INTERVAL"
)
SFT_ARGS=(
  --rollout-function-path vime.rollout.sft_rollout.generate_rollout
  --prompt-data "$SFT_PROMPT_DATA"
  --input-key messages
  --rollout-shuffle
  --num-epoch "$SFT_NUM_EPOCH"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)
PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${SFT_LR:-1e-5}"
  --lr-decay-style cosine
  --min-lr "${SFT_MIN_LR:-1e-6}"
  --lr-warmup-fraction "${SFT_WARMUP_FRACTION:-0.03}"
  --weight-decay "${SFT_WEIGHT_DECAY:-0.1}"
  --adam-beta1 0.9
  --adam-beta2 0.98
  --use-distributed-optimizer
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

set -x
"$TRAIN_ENV/bin/ray" job submit --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
  --runtime-env-json="$RUNTIME_ENV_JSON" -- \
  "$TRAIN_ENV/bin/python" "$VIME_ROOT/train_async.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$ACTOR_GPUS" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${MISC_ARGS[@]}" 2>&1 | tee "$RUN_DIR/launch.log"
