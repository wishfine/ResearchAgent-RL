#!/usr/bin/env bash
# Frozen-policy clients only: never start/stop GPU servers, Ray or training.
set -euo pipefail
PROJECT="${PROJECT:-/local_data/$USER/ResearchAgent-RL}"
BASE="${BASE:-/local_data/$USER/research-agent-rl-data}"
PY="${PY:-/local_data/$USER/conda_envs/research-agent-runtime/bin/python}"
COMPARE_ROOT="${COMPARE_ROOT:-$BASE/runtime_efficiency/musique_retriever_compare24_$(date +%Y%m%d_%H%M%S)}"
QWEN_INDEX="${QWEN_INDEX:-$BASE/dense_indexes/musique_train_qwen3_embed_06b}"
QWEN_URL="${QWEN_URL:-http://127.0.0.1:8107/v1}"
MODEL="${MODEL:-/local_data/$USER/models/ResearchAgent-Qwen3.5-9B-SFT874}"
cd "$PROJECT"
test -x "$PY"
test -f "$QWEN_INDEX/manifest.json"
mkdir -p "$COMPARE_ROOT" "$PROJECT/artifacts/runtime_efficiency" "$BASE/runtime_efficiency"
exec 9>"$COMPARE_ROOT/compare.lock"
flock -n 9 || { echo "[ERROR] Another driver owns $COMPARE_ROOT" >&2; exit 2; }
LINK="$PROJECT/artifacts/runtime_efficiency/$(basename "$COMPARE_ROOT")"
if [ -e "$LINK" ] || [ -L "$LINK" ]; then
  test "$(readlink -f "$LINK")" = "$(readlink -f "$COMPARE_ROOT")" || {
    echo "[ERROR] Refusing to overwrite $LINK" >&2; exit 2;
  }
else
  ln -s "$COMPARE_ROOT" "$LINK"
fi
printf '%s\n' "$COMPARE_ROOT" > "$BASE/runtime_efficiency/latest_retriever_compare.txt"
echo "Frozen SFT874 retriever diagnostic: same tasks, seeds, sampling and four prompt/READ arms."
echo "COMPARE_ROOT=$COMPARE_ROOT"
echo "Defaults: 24 tasks x 2 repeats x 4 arms x 2 retrievers = 384 episodes; not RL."

COMMON_ARGS=(--tasks_dir "$PROJECT/data/musique_rl_v2/tasks/train"
  --corpus_dir "$PROJECT/data/musique_rl_v2/corpus/train"
  --model_url "${MODEL_URL:-http://127.0.0.1:8105/v1}"
  --model_name "${MODEL_NAME:-Qwen3.5-9B-SFT874}" --model_dir "$MODEL"
  --per_hop "${PER_HOP:-8}" --repeats "${REPEATS:-2}"
  --selection_seed "${SELECTION_SEED:-20260929}"
  --max_steps 15 --max_tokens 512 --temperature 0.7 --top_p 0.95
  --max_api_calls "${MAX_API_CALLS:-2880}"
  --max_generated_tokens "${MAX_GENERATED_TOKENS:-1474560}"
  --max_wall_sec "${MAX_WALL_SEC:-14400}")

run_client() {
  local name="$1"
  shift
  local mode=bm25
  [ "$name" = qwen_hybrid ] && mode=hybrid
  local extra=()
  [ "$mode" = hybrid ] && extra=(--dense_index "$QWEN_INDEX" --embedding_url "$QWEN_URL")
  # Call the existing Python driver directly: no global bm25/qwen_hybrid symlinks.
  CUDA_VISIBLE_DEVICES="" "$PY" -u scripts/run_musique_protocol_smoke.py \
    "${COMMON_ARGS[@]}" --output_dir "$COMPARE_ROOT/$name" --retrieval "$mode" \
    "${extra[@]}" "$@"
}

# Prepare both immutable manifests and compare before spending any model calls.
for name in bm25 qwen_hybrid; do
  mkdir -p "$COMPARE_ROOT/$name"
  if run_client "$name" --prepare_only > "$COMPARE_ROOT/$name/prepare.log" 2>&1; then
    echo "$name manifest prepared"
  else
    status=$?
    tail -n 30 "$COMPARE_ROOT/$name/prepare.log" >&2
    exit "$status"
  fi
done
"$PY" scripts/summarize_musique_retrievers.py --root "$COMPARE_ROOT" --validate_only
if [ "${PREPARE_ONLY:-0}" = 1 ]; then
  echo "RETRIEVER_COMPARE_PREPARED (no model calls)"
  exit 0
fi

# Validate live APIs, local model identity and full index checksums before BM25 calls.
"$PY" scripts/summarize_musique_retrievers.py --root "$COMPARE_ROOT" --preflight

# Serial to avoid artificial contention on the policy/embedding services.
for name in bm25 qwen_hybrid; do
  echo "$(date -Iseconds) | $name | start"
  if run_client "$name" >> "$COMPARE_ROOT/$name/driver.log" 2>&1; then
    echo "$(date -Iseconds) | $name | completed"
  else
    status=$?
    echo "[ERROR] $name exited $status; inspect $COMPARE_ROOT/$name/driver.log" >&2
    "$PY" scripts/summarize_musique_retrievers.py --root "$COMPARE_ROOT" || true
    exit "$status"
  fi
done
"$PY" scripts/summarize_musique_retrievers.py --root "$COMPARE_ROOT" --require_complete
echo "RETRIEVER_COMPARE_COMPLETED"
