#!/usr/bin/env bash
# Read existing model only. No pip install, no downloads, no process killing.
set -euo pipefail
FAMILY="${1:-bge}"
BASE="${BASE:-/local_data/$USER/research-agent-rl-data}"
ENV="${ENV:-/local_data/$USER/conda_envs/research-agent-runtime}"
if [ "$FAMILY" = "bge" ]; then
  MODEL="${EMBED_MODEL_DIR:-/local_data/$USER/data/bio-know-tag/models/bge-small-zh-v1.5}"
  NAME="${EMBED_MODEL_NAME:-bge-small-zh-v1.5}"
  POOLING=CLS
  GPU="${EMBED_GPU:-1}"
  PORT="${EMBED_PORT:-8106}"
  CONTEXT=512
elif [ "$FAMILY" = "qwen" ]; then
  MODEL="${EMBED_MODEL_DIR:-/local_data/$USER/models/Qwen3-Embedding-0.6B}"
  NAME="${EMBED_MODEL_NAME:-Qwen3-Embedding-0.6B}"
  POOLING=LAST
  GPU="${EMBED_GPU:-2}"
  PORT="${EMBED_PORT:-8107}"
  CONTEXT=4096
else
  echo "Usage: bash scripts/start_embedding_server.sh bge|qwen" >&2
  exit 2
fi
test -f "$MODEL/config.json"
MODEL="$(readlink -f "$MODEL")"
test -x "$ENV/bin/python"
BUSY="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)"
if [ -n "$BUSY" ]; then
  echo "[ERROR] GPU $GPU is occupied: $BUSY. Choose a free EMBED_GPU." >&2
  exit 2
fi
if ss -ltnH | awk '{print $4}' | grep -Eq ":${PORT}$"; then
  echo "[ERROR] Port $PORT already occupied; no process stopped." >&2
  exit 2
fi
SITE="$ENV/lib/python3.11/site-packages"
NVIDIA_LIBS="$(find "$SITE/nvidia" -type d -name lib -print | paste -sd: -)"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$BASE/hf_cache" TMPDIR="$BASE/runtime_efficiency/tmp"
mkdir -p "$TMPDIR"
SERVER_DIR="$BASE/runtime_efficiency/server_embed_${FAMILY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SERVER_DIR"
printf '%s\n' "model=$MODEL" "served_name=$NAME" "gpu=$GPU" "port=$PORT" \
  "pooling=$POOLING" "context=$CONTEXT" > "$SERVER_DIR/run_info.txt"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$ENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" \
  --runner pooling --convert embed \
  --pooler-config "{\"task\":\"embed\",\"pooling_type\":\"$POOLING\",\"use_activation\":true}" \
  --dtype half --max-model-len "$CONTEXT" --gpu-memory-utilization 0.10 \
  --max-num-seqs 32 --max-num-batched-tokens 16384 --enforce-eager \
  > "$SERVER_DIR/server.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$SERVER_DIR/pid.txt"
printf '%s\n' "$SERVER_DIR" > "$BASE/runtime_efficiency/latest_embedding_${FAMILY}.txt"
echo "SERVER_DIR=$SERVER_DIR PID=$PID GPU=$GPU PORT=$PORT"
echo "Wait for /v1/models, then run scripts/verify_embedding_backend.py before indexing."
