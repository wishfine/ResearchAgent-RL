#!/usr/bin/env bash
# Shortest MuSiQue GRPO baseline path; no embedding service dependency.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BASE="${BASE:-/local_data/$USER/research-agent-rl-data}"
export TRAIN_ENV="${TRAIN_ENV:-/local_data/$USER/conda_envs/research-agent-runtime}"
export VIME_ROOT="${VIME_ROOT:-/local_data/$USER/vime-src/vime}"
export MEGATRON_ROOT="${MEGATRON_ROOT:-/local_data/$USER/vime-src/Megatron-LM}"
export CUDA_HOME="${CUDA_HOME:-/local_data/$USER/cuda-toolkit-12.9/usr/local/cuda-12.9}"
export PATH="$TRAIN_ENV/bin:$CUDA_HOME/bin:$PATH"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/local_data/$USER/models/ResearchAgent-Qwen3.5-9B-SFT874}"
export SFT_CHECKPOINTS="${SFT_CHECKPOINTS:-$BASE/checkpoints/sft874}"
export CORPUS_DIR="${CORPUS_DIR:-$PROJECT_ROOT/data/musique_rl_v2/corpus/train}"
export PROMPT_DATA="${PROMPT_DATA:-$BASE/slime_data/musique_train.jsonl}"
export GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export RESEARCH_AGENT_READ_MODE="${RESEARCH_AGENT_READ_MODE:-full}"
export RESEARCH_AGENT_PROMPT_MODE="${RESEARCH_AGENT_PROMPT_MODE:-adaptive}"
export RESEARCH_AGENT_MAX_STEPS="${RESEARCH_AGENT_MAX_STEPS:-15}"
export RESEARCH_AGENT_STEP_PENALTY="${RESEARCH_AGENT_STEP_PENALTY:-0.005}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export RAY_PORT="${RAY_PORT:-6398}" RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8298}"
export RAY_TMPDIR="${RAY_TMPDIR:-/local_data/$USER/r}"
# Old launcher uses global `ray stop` on exit. Never do that silently on35.
export RAY_AUTO_STOP="${RAY_AUTO_STOP:-0}"
export RUN_DIR="${RUN_DIR:-$BASE/outputs/vime_musique_grpo_$(date +%Y%m%d_%H%M%S)}"
echo "ACTOR_GPUS=$ACTOR_GPUS ROLLOUT_GPUS=$ROLLOUT_GPUS GPU_IDS=$GPU_IDS"
echo "NUM_ROLLOUT=$NUM_ROLLOUT N_SAMPLES_PER_PROMPT=$N_SAMPLES_PER_PROMPT"
echo "READ_MODE=$RESEARCH_AGENT_READ_MODE PROMPT_MODE=$RESEARCH_AGENT_PROMPT_MODE ATTENTION_BACKEND=$ATTENTION_BACKEND"
echo "RUN_DIR=$RUN_DIR SFT_CHECKPOINTS=$SFT_CHECKPOINTS"
if [ "${1:-}" = "--print-config" ]; then exit 0; fi
test -f "$VIME_ROOT/train_async.py"
test -f "$MEGATRON_ROOT/megatron/core/__init__.py"
test -f "$HF_CHECKPOINT/config.json"
test -f "$CORPUS_DIR/corpus.sqlite"
test -f "$SFT_CHECKPOINTS/iter_0000874/metadata.json"
test -f "$SFT_CHECKPOINTS/latest_checkpointed_iteration.txt"
if [ ! -f "$PROMPT_DATA" ]; then
  "$TRAIN_ENV/bin/python" "$PROJECT_ROOT/scripts/prepare_slime_dataset.py" \
    --tasks_dir "$PROJECT_ROOT/data/musique_rl_v2/tasks/train" \
    --output_file "$PROMPT_DATA"
fi
"$TRAIN_ENV/bin/python" - "$PROJECT_ROOT" "$PROMPT_DATA" <<'PY'
import json
from pathlib import Path
import sys
root, prompt = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from scripts.prepare_slime_dataset import build_records
expected, _ = build_records(root / "data/musique_rl_v2/tasks/train")
with prompt.open() as handle:
    actual = [json.loads(line) for line in handle if line.strip()]
if actual != expected:
    raise SystemExit("Existing MuSiQue prompt data does not match current train tasks. Use a fresh PROMPT_DATA path; old data was not overwritten.")
print("MuSiQue train prompt provenance: OK", len(actual))
PY
exec bash "$PROJECT_ROOT/scripts/run_vime_hotpotqa_sft_grpo_v2.sh"
