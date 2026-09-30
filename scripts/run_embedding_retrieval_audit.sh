#!/usr/bin/env bash
# CPU lexical/vector retrieval + requests to already-running embedding services.
set -euo pipefail
PROJECT="${PROJECT:-/local_data/$USER/ResearchAgent-RL}"
BASE="${BASE:-/local_data/$USER/research-agent-rl-data}"
PY="${PY:-/local_data/$USER/conda_envs/research-agent-runtime/bin/python}"
RUN_DIR="${RUN_DIR:-$BASE/runtime_efficiency/embedding_retrieval_audit_$(date +%Y%m%d_%H%M%S)}"
MODELS="${EMBED_AUDIT_MODELS:-bge qwen}"
mkdir -p "$RUN_DIR"
cd "$PROJECT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
test -f data/musique_rl_v2/corpus/train/corpus.sqlite
COMMON=(--tasks_dir "$PROJECT/data/musique_rl_v2/tasks/train"
        --corpus_dir "$PROJECT/data/musique_rl_v2/corpus/train"
        --topk 5 10 20 --max_tasks "${AUDIT_TASKS:-1000}" --seed 42)
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
"$PY" -u scripts/evaluate_retrieval.py "${COMMON[@]}" --retrieval bm25 \
  --output_file "$RUN_DIR/bm25.json" > "$RUN_DIR/bm25.log" 2>&1
for FAMILY in $MODELS; do
  if [ "$FAMILY" = "bge" ]; then
    INDEX="${BGE_INDEX:-$BASE/dense_indexes/musique_train_bge_small_zh_v15}"
    URL="${BGE_URL:-http://127.0.0.1:8106/v1}"
  elif [ "$FAMILY" = "qwen" ]; then
    INDEX="${QWEN_INDEX:-$BASE/dense_indexes/musique_train_qwen3_embed_06b}"
    URL="${QWEN_URL:-http://127.0.0.1:8107/v1}"
  else
    echo "[ERROR] Unknown embedding family: $FAMILY" >&2
    exit 2
  fi
  test -f "$INDEX/manifest.json"
  for MODE in dense hybrid; do
    echo "Audit: $FAMILY $MODE"
    OUT="$RUN_DIR/${FAMILY}_${MODE}.json"
    if [ -f "$OUT" ] || [ -f "$OUT.embedding_cost.jsonl" ]; then
      echo "[ERROR] Existing audit output: $OUT. Use new RUN_DIR." >&2
      exit 2
    fi
    "$PY" -u scripts/evaluate_retrieval.py "${COMMON[@]}" \
      --retrieval "$MODE" --dense_index "$INDEX" --embedding_url "$URL" \
      --output_file "$OUT" > "$RUN_DIR/${FAMILY}_${MODE}.log" 2>&1
  done
done
echo "RETRIEVAL_AUDIT_COMPLETED RUN_DIR=$RUN_DIR"
