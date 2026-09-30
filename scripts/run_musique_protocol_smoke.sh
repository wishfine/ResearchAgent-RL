#!/usr/bin/env bash
# API-only client. Does not start/stop a server or occupy another GPU.
set -euo pipefail
PROJECT="${PROJECT:-/local_data/$USER/ResearchAgent-RL}"
BASE="${BASE:-/local_data/$USER/research-agent-rl-data}"
PY="${PY:-/local_data/$USER/conda_envs/research-agent-runtime/bin/python}"
MODEL="${MODEL:-/local_data/$USER/models/ResearchAgent-Qwen3.5-9B-SFT874}"
RUN_DIR="${RUN_DIR:-$BASE/runtime_efficiency/musique_protocol24_$(date +%Y%m%d_%H%M%S)}"
cd "$PROJECT"
test -x "$PY"
test -f "$MODEL/config.json"
test -f "$PROJECT/data/musique_rl_v2/corpus/train/corpus.sqlite"
mkdir -p "$RUN_DIR" "$PROJECT/artifacts/runtime_efficiency" "$BASE/runtime_efficiency"
LINK="$PROJECT/artifacts/runtime_efficiency/$(basename "$RUN_DIR")"
if [ -e "$LINK" ] || [ -L "$LINK" ]; then
  if [ "$(readlink -f "$LINK")" != "$(readlink -f "$RUN_DIR")" ]; then
    echo "[ERROR] Refusing to overwrite $LINK" >&2
    exit 2
  fi
else
  ln -s "$RUN_DIR" "$LINK"
fi
printf '%s\n' "$RUN_DIR" > "$BASE/runtime_efficiency/latest_protocol_smoke.txt"
echo "RUN_DIR=$RUN_DIR"
RETRIEVAL="${RETRIEVAL:-bm25}"
EXTRA_ARGS=(--retrieval "$RETRIEVAL")
if [ "$RETRIEVAL" != "bm25" ]; then
  DENSE_INDEX="${DENSE_INDEX:-$BASE/dense_indexes/musique_train_bge_small_zh_v15}"
  test -f "$DENSE_INDEX/manifest.json"
  EXTRA_ARGS+=(--dense_index "$DENSE_INDEX" --embedding_url "${EMBEDDING_URL:-http://127.0.0.1:8106/v1}")
fi
exec "$PY" -u scripts/run_musique_protocol_smoke.py \
  --tasks_dir "$PROJECT/data/musique_rl_v2/tasks/train" \
  --corpus_dir "$PROJECT/data/musique_rl_v2/corpus/train" \
  --output_dir "$RUN_DIR" \
  --model_url "${MODEL_URL:-http://127.0.0.1:8105/v1}" \
  --model_name "${MODEL_NAME:-Qwen3.5-9B-SFT874}" \
  --model_dir "$MODEL" \
  --per_hop "${PER_HOP:-8}" --repeats "${REPEATS:-2}" \
  --selection_seed "${SELECTION_SEED:-20260929}" \
  --max_steps 15 --max_tokens 512 --temperature 0.7 --top_p 0.95 \
  --max_api_calls "${MAX_API_CALLS:-2880}" \
  --max_generated_tokens "${MAX_GENERATED_TOKENS:-1474560}" \
  --max_wall_sec "${MAX_WALL_SEC:-14400}" "${EXTRA_ARGS[@]}" "$@"
