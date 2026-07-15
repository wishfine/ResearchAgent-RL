#!/usr/bin/env bash
# Six-GPU Vime fully-async smoke run for ResearchAgent-RL on HotpotQA.
#
# Default allocation: GPUs 0-1 are left untouched; GPUs 2-5 run the actor
# (TP=2, DP=2) and GPUs 6-7 run one rollout engine (TP=2).
#
# Prerequisites:
# - Vime source and Megatron source installed under /data/$USER/vime-src.
# - Qwen3.5-9B converted with tools/convert_hf_to_torch_dist.py.
# - This repository is checked out on the node running Ray.

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
PROMPT_DATA="${PROMPT_DATA:-$BASE/slime_data/hotpotqa_train_7k.jsonl}"
CORPUS_DIR="${CORPUS_DIR:-$BASE/hotpotqa_7k3k/corpus/train}"
RUN_DIR="${RUN_DIR:-$BASE/outputs/vime_hotpotqa_smoke6_$(date +%Y%m%d_%H%M%S)}"

GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

IFS=',' read -r -a GPU_LIST <<<"$GPU_IDS"
[[ ${#GPU_LIST[@]} -eq 6 ]] || fail "GPU_IDS must contain exactly six GPU IDs, got: $GPU_IDS"
[[ $((ACTOR_GPUS + ROLLOUT_GPUS)) -eq 6 ]] || fail "ACTOR_GPUS + ROLLOUT_GPUS must equal 6"
[[ -d "$VIME_ROOT" ]] || fail "Vime source not found: $VIME_ROOT"
[[ -d "$MEGATRON_ROOT" ]] || fail "Megatron source not found: $MEGATRON_ROOT"
[[ -d "$REF_CHECKPOINT" ]] || fail "Converted checkpoint not found: $REF_CHECKPOINT"
[[ -f "$PROMPT_DATA" ]] || fail "Prompt data not found: $PROMPT_DATA"
[[ -d "$CORPUS_DIR" ]] || fail "Corpus not found: $CORPUS_DIR"
[[ -x "$TRAIN_ENV/bin/python" ]] || fail "Training Python not found: $TRAIN_ENV/bin/python"

"$TRAIN_ENV/bin/python" - "$PROMPT_DATA" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    first_record = json.loads(next(line for line in handle if line.strip()))
prompt = first_record.get("prompt")
if not (
    isinstance(prompt, list)
    and prompt
    and isinstance(prompt[0], dict)
    and prompt[0].get("role") == "user"
    and isinstance(prompt[0].get("content"), str)
):
    raise SystemExit(
        "Vime/Qwen3.5 requires a chat-message prompt. Regenerate the dataset with "
        "scripts/prepare_slime_dataset.py from commit a03558f or newer."
    )
print("prompt dataset format: Vime chat-message OK")
PY

for gpu in "${GPU_LIST[@]}"; do
  gpu_processes="$(
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk 'NF'
  )"
  [[ -z "$gpu_processes" ]] || fail "GPU $gpu is busy (PID(s): $gpu_processes). Stop or relocate that workload before launching Vime."
done

NVIDIA_LIBRARY_PATH="$(find "$SITE/nvidia" -type d -name lib -print | paste -sd: -)"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_HOME CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT:$MEGATRON_ROOT:$VIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Ray embeds its session name below this directory in Unix-socket paths.  Keep
# it short: the kernel limit is 107 bytes and "$BASE/ray" is too long here.
export RAY_TMPDIR="${RAY_TMPDIR:-/data/$USER/ray}"
export RESEARCH_AGENT_CORPUS_DIR="$CORPUS_DIR"
export RESEARCH_AGENT_MAX_STEPS="${RESEARCH_AGENT_MAX_STEPS:-6}"

mkdir -p "$RUN_DIR" "$RAY_TMPDIR"

"$TRAIN_ENV/bin/python" - <<'PY'
import aiohttp_cors
import opencensus
import opentelemetry.exporter.prometheus
import ray
import transformer_engine.pytorch
import vime
from torch_memory_saver import torch_memory_saver

print("preflight imports: OK")
print("ray:", ray.__version__)
print("vime:", vime.__file__)
print("torch_memory_saver:", torch_memory_saver.__class__.__module__)
PY

if "$TRAIN_ENV/bin/ray" status --address="127.0.0.1:$RAY_PORT" >/dev/null 2>&1; then
  fail "A Ray cluster is already listening on port $RAY_PORT. Stop only the intended cluster first: $TRAIN_ENV/bin/ray stop --force"
fi

cd "$VIME_ROOT"
# Shell array supplied by Vime for the exact Qwen3.5-9B architecture.
source scripts/models/qwen3.5-9B.sh

"$TRAIN_ENV/bin/ray" start --head \
  --node-ip-address 127.0.0.1 \
  --port "$RAY_PORT" \
  --dashboard-port "$RAY_DASHBOARD_PORT" \
  --num-gpus 6 \
  --disable-usage-stats

# `ray start` returns before the dashboard job-submission endpoint is always
# accepting connections.  Wait rather than racing `ray job submit` below.
dashboard_ready=0
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$RAY_DASHBOARD_PORT/api/version" >/dev/null; then
    dashboard_ready=1
    break
  fi
  sleep 1
done
if [[ "$dashboard_ready" -ne 1 ]]; then
  echo "Ray dashboard did not become ready within 60 seconds." >&2
  find "$RAY_TMPDIR/ray/session_latest/logs" -maxdepth 1 -type f \
    \( -name 'dashboard*.log' -o -name 'dashboard*.err' \) -print -exec tail -n 80 {} \; \
    2>/dev/null || true
  exit 1
fi

RUNTIME_ENV_JSON="$(
  "$TRAIN_ENV/bin/python" - <<PY
import json
import os

print(json.dumps({"env_vars": {
    "PYTHONPATH": os.environ["PYTHONPATH"],
    "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
    "CUDA_HOME": os.environ["CUDA_HOME"],
    "CUDA_PATH": os.environ["CUDA_PATH"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "RESEARCH_AGENT_CORPUS_DIR": os.environ["RESEARCH_AGENT_CORPUS_DIR"],
    "RESEARCH_AGENT_MAX_STEPS": os.environ["RESEARCH_AGENT_MAX_STEPS"],
}}))
PY
)"

CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_CHECKPOINT"
  --save "$RUN_DIR/checkpoints"
  --save-interval 9999
)

ROLLOUT_ARGS=(
  --rollout-function-path vime.rollout.fully_async_rollout.generate_rollout_fully_async
  --custom-generate-function-path research_agent.adapters.vime.custom_rollout.custom_generate
  --custom-rm-path research_agent.adapters.vime.custom_rollout.custom_rm
  --prompt-data "$PROMPT_DATA"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout 1
  --rollout-batch-size 1
  --n-samples-per-prompt 2
  --rollout-max-response-len 256
  --rollout-temperature 1.0
  --micro-batch-size 1
  --global-batch-size 2
  --balance-data
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
  --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

VLLM_ARGS=(
  --rollout-num-gpus-per-engine 2
  --vllm-gpu-memory-utilization 0.70
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
"$TRAIN_ENV/bin/ray" job submit \
  --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- \
  "$TRAIN_ENV/bin/python" "$VIME_ROOT/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${VLLM_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/launch.log"
