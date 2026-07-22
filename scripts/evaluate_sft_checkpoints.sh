#!/usr/bin/env bash
# Sequentially materialize and evaluate Vime SFT checkpoints on a fixed
# HotpotQA sample. All defaults can be overridden with environment variables.

set -euo pipefail

PROJECT="${PROJECT:-$HOME/ResearchAgent-RL}"
BASE="${BASE:-/data/$USER/research-agent-rl-data}"
TRAIN_ENV="${TRAIN_ENV:-/data/$USER/conda_envs/vime-train-cu129}"
TRAIN_PY="${TRAIN_PY:-$TRAIN_ENV/bin/python}"
TRAIN_VLLM="${TRAIN_VLLM:-$TRAIN_ENV/bin/vllm}"
EVAL_PY="${EVAL_PY:-$HOME/miniconda3/envs/vime-runtime/bin/python}"

VIME_ROOT="${VIME_ROOT:-/data/$USER/vime-src/vime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/data/$USER/vime-src/Megatron-LM}"
CUDA_HOME="${CUDA_HOME:-/data/$USER/cuda-toolkit-12.9/usr/local/cuda-12.9}"
BASE_MODEL="${BASE_MODEL:-/home/$USER/models/models/Qwen--Qwen3.5-9B/snapshots/master}"
SFT_RUN="${SFT_RUN:-$BASE/outputs/vime_hotpotqa_sft7k_20260721_162728}"
TASKS_DIR="${TASKS_DIR:-$BASE/hotpotqa_7k3k/tasks/eval}"
CORPUS_DIR="${CORPUS_DIR:-$BASE/hotpotqa_7k3k/corpus/eval}"

CHECKPOINTS="${CHECKPOINTS:-0000249 0000499 0000749 0000874}"
VLLM_GPU="${VLLM_GPU:-2}"
VLLM_PORT="${VLLM_PORT:-8101}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_EPISODES="${MAX_EPISODES:-100}"
SELECTION_SEED="${SELECTION_SEED:-20260713}"
MAX_STEPS="${MAX_STEPS:-6}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT/results/hotpotqa_eval_sft_ckpts_n${MAX_EPISODES}_$(date +%Y%m%d_%H%M%S)}"

for executable in "$TRAIN_PY" "$TRAIN_VLLM" "$EVAL_PY"; do
  if [[ ! -x "$executable" ]]; then
    echo "[ERROR] Required executable is unavailable: $executable" >&2
    exit 1
  fi
done

for path in "$PROJECT/scripts/materialize_vime_hf_checkpoint.py" \
  "$VIME_ROOT/tools/convert_torch_dist_to_hf.py" "$BASE_MODEL" \
  "$TASKS_DIR" "$CORPUS_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Required path is unavailable: $path" >&2
    exit 1
  fi
done

SITE="$TRAIN_ENV/lib/python3.11/site-packages"
NVIDIA_LIBRARY_PATH="$(find "$SITE/nvidia" -type d -name lib -print | paste -sd: -)"
export PYTHONPATH="$MEGATRON_ROOT:$VIME_ROOT:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$EVAL_ROOT"
printf '%s\n' "$EVAL_ROOT" > "$EVAL_ROOT/run_dir.txt"

server_pid=""
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT INT TERM

wait_for_port_release() {
  for _ in $(seq 1 30); do
    if ! ss -ltn | grep -q ":${VLLM_PORT} "; then
      return 0
    fi
    sleep 2
  done
  echo "[ERROR] Port $VLLM_PORT is still occupied." >&2
  return 1
}

wait_for_server() {
  local models_file="$1"
  local vllm_log="$2"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >"$models_file" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[ERROR] vLLM exited before readiness." >&2
      tail -n 120 "$vllm_log" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "[ERROR] Timed out waiting for vLLM on port $VLLM_PORT." >&2
  tail -n 120 "$vllm_log" >&2 || true
  return 1
}

echo "EVAL_ROOT=$EVAL_ROOT"
echo "CHECKPOINTS=$CHECKPOINTS"
echo "VLLM_GPU=$VLLM_GPU VLLM_PORT=$VLLM_PORT"

for iteration in $CHECKPOINTS; do
  checkpoint="$SFT_RUN/checkpoints/iter_$iteration"
  run_dir="$EVAL_ROOT/iter_$iteration"
  hf_partial="$run_dir/hf_partial"
  hf_merged="$run_dir/hf_merged"
  model_name="Qwen3.5-9B-SFT-$iteration"

  if [[ ! -d "$checkpoint" ]]; then
    echo "[ERROR] Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ -e "$run_dir/metrics_summary.json" ]]; then
    echo "[ERROR] Refusing to overwrite completed result: $run_dir/metrics_summary.json" >&2
    exit 1
  fi
  mkdir -p "$run_dir"

  echo "========== $(date '+%F %T') | iter_$iteration | convert =========="
  "$TRAIN_PY" "$VIME_ROOT/tools/convert_torch_dist_to_hf.py" \
    --input-dir "$checkpoint/" \
    --output-dir "$hf_partial" \
    --origin-hf-dir "$BASE_MODEL" \
    >"$run_dir/convert.log" 2>&1

  echo "========== $(date '+%F %T') | iter_$iteration | materialize =========="
  "$TRAIN_PY" "$PROJECT/scripts/materialize_vime_hf_checkpoint.py" \
    --base_model_dir "$BASE_MODEL" \
    --trained_weights_dir "$hf_partial" \
    --output_dir "$hf_merged" \
    --overwrite \
    >"$run_dir/materialize.log" 2>&1

  wait_for_port_release
  echo "========== $(date '+%F %T') | iter_$iteration | vLLM start =========="
  CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
  "$TRAIN_VLLM" serve "$hf_merged" \
    --served-model-name "$model_name" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    >"$run_dir/vllm.log" 2>&1 < /dev/null &
  server_pid=$!
  printf '%s\n' "$server_pid" > "$run_dir/vllm.pid"

  wait_for_server "$run_dir/models.json" "$run_dir/vllm.log"

  echo "========== $(date '+%F %T') | iter_$iteration | eval n=$MAX_EPISODES =========="
  "$EVAL_PY" "$PROJECT/scripts/run_llm_baseline.py" \
    --model_url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --model_name "$model_name" \
    --tasks_dir "$TASKS_DIR" \
    --corpus_dir "$CORPUS_DIR" \
    --max_episodes "$MAX_EPISODES" \
    --selection_seed "$SELECTION_SEED" \
    --max_steps "$MAX_STEPS" \
    --api_ready_timeout_sec 0 \
    --output_dir "$run_dir/eval" \
    >"$run_dir/eval.log" 2>&1

  cp "$run_dir/eval/metrics_summary.json" "$run_dir/metrics_summary.json"
  cleanup_server
  echo "========== $(date '+%F %T') | iter_$iteration | completed =========="
done

echo "ALL_FOUR_CHECKPOINTS_COMPLETED"
