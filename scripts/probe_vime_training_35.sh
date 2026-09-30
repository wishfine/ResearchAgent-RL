#!/usr/bin/env bash
# Read-only training readiness probe. Does not start Ray, allocate GPUs or install packages.
set -u
PROJECT="${PROJECT:-/local_data/$USER/ResearchAgent-RL}"
BASE="${BASE:-/local_data/$USER/research-agent-rl-data}"
ENV="${TRAIN_ENV:-/local_data/$USER/conda_envs/research-agent-runtime}"
VIME_ROOT="${VIME_ROOT:-/local_data/$USER/vime-src/vime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/local_data/$USER/vime-src/Megatron-LM}"
CUDA_HOME="${CUDA_HOME:-/local_data/$USER/cuda-toolkit-12.9}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/local_data/$USER/models/ResearchAgent-Qwen3.5-9B-SFT874}"
export VIME_ROOT MEGATRON_ROOT CUDA_HOME HF_CHECKPOINT
export PATH="$ENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PROJECT:$MEGATRON_ROOT:$VIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SITE="$ENV/lib/python3.11/site-packages"
NVIDIA_LIBS="$(find "$SITE/nvidia" -type d -name lib -print 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ENV/bin/python" - "$PROJECT" "$BASE" <<'PY'
from pathlib import Path
import importlib
import os
import sys
import traceback
print("python:", sys.executable)
project, base = map(Path, sys.argv[1:])
required = [Path(os.environ["VIME_ROOT"]) / "train_async.py",
            Path(os.environ["MEGATRON_ROOT"]) / "megatron/core/__init__.py",
            base / "checkpoints/sft874/iter_0000874/metadata.json",
            base / "checkpoints/sft874/latest_checkpointed_iteration.txt",
            project / "data/musique_rl_v2/corpus/train/corpus.sqlite",
            Path(os.environ["HF_CHECKPOINT"]) / "config.json",
            Path(os.environ["CUDA_HOME"]) / "bin/nvcc"]
errors = []
for path in required:
    exists = path.is_file()
    print("PATH", "OK" if exists else "MISSING", path)
    if not exists:
        errors.append(str(path))
for name in ("torch", "vllm", "ray", "transformer_engine.pytorch", "fla.modules",
             "mbridge", "torch_memory_saver", "aiohttp_cors", "opencensus",
             "opentelemetry.exporter.prometheus", "pylatexenc", "megatron.core",
             "vime.rollout.fully_async_rollout", "research_agent.adapters.vime.custom_rollout"):
    try:
        module = importlib.import_module(name)
        print("IMPORT OK", name, getattr(module, "__file__", None))
    except Exception:
        errors.append(name)
        print("IMPORT FAIL", name)
        traceback.print_exc(limit=3)
try:
    from megatron.core.transformer.enums import AttnBackend
    assert "auto" in AttnBackend.__members__
    print("TE attention backend auto: supported")
except Exception:
    errors.append("attention backend auto")
    traceback.print_exc(limit=3)
print("TRAIN_IMPORTS_READY" if not errors else "TRAIN_IMPORTS_NOT_READY", errors)
print("Import readiness is not a GPU training/optimizer/weight-sync success claim.")
raise SystemExit(bool(errors))
PY
STATUS=$?
nvidia-smi --query-gpu=index,name,memory.used --format=csv
exit "$STATUS"
