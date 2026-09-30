#!/usr/bin/env bash
# Read-only inventory for moving the frozen-policy experiment from 45 to 35.
# Run with bash; this script does not install packages or start GPU workloads.
set -uo pipefail

PROBE_USER="$(id -un)"
PROBE_BASE="${PROBE_BASE:-/data/$PROBE_USER/research-agent-rl-data}"
PROBE_ENV="${PROBE_ENV:-/data/$PROBE_USER/conda_envs/vime-train-cu129}"
PROBE_PROJECT="${PROBE_PROJECT:-$HOME/ResearchAgent-RL}"

printf 'host=%s\nuser=%s\ntime=%s\n' "$(hostname)" "$PROBE_USER" "$(date -Is 2>/dev/null || date)"
uname -sm
getconf GNU_LIBC_VERSION 2>/dev/null || true
df -h /data "$HOME" 2>/dev/null || true
if command -v findmnt >/dev/null 2>&1; then
  findmnt -T /data -o TARGET,SOURCE,FSTYPE,OPTIONS || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total --format=csv,noheader || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
fi
for executable in rsync conda conda-pack nvcc; do
  command -v "$executable" || true
done

for prefix in "$PROBE_ENV" "$HOME/miniconda3/envs/vime-runtime"; do
  printf '\nenvironment_prefix=%s\n' "$prefix"
  if [[ ! -x "$prefix/bin/python" ]]; then
    printf 'python=UNAVAILABLE\n'
    continue
  fi
  "$prefix/bin/python" - <<'PY'
import importlib.metadata as md
import importlib.util as iu
import json
import platform
import sys

print('python:', sys.executable, platform.python_version())
for package in ('torch', 'vllm', 'numpy', 'transformers', 'ray', 'vime',
                'mbridge', 'flash-attn', 'transformer-engine', 'conda-pack',
                'pydantic', 'rank-bm25', 'pytest'):
    try:
        print(package + ':', md.version(package))
    except md.PackageNotFoundError:
        print(package + ': MISSING')
for package in ('vime', 'mbridge', 'megatron-core'):
    try:
        value = json.loads(md.distribution(package).read_text('direct_url.json') or '{}')
        # Report only editable local paths, never authenticated repository URLs.
        if value.get('dir_info', {}).get('editable'):
            url = value.get('url', '')
            print('editable_source:', package, url if url.startswith('file:') else 'NONLOCAL')
    except (md.PackageNotFoundError, ValueError):
        pass
try:
    import torch
    print('torch_build:', torch.__version__, 'cuda:', torch.version.cuda)
    print('cuda_available:', torch.cuda.is_available())
except Exception as exc:
    print('torch_import_error:', type(exc).__name__, str(exc))
for module in ('conda_pack', 'vime', 'megatron', 'mbridge'):
    try:
        spec = iu.find_spec(module)
        print('module_path:', module, spec.origin if spec else 'MISSING')
    except Exception as exc:
        print('module_lookup_error:', module, type(exc).__name__, str(exc))
PY
done

for directory in "$PROBE_PROJECT" "$PROBE_BASE/musique_rl_v2" \
  "/data/$PROBE_USER/vime-src/vime" "/data/$PROBE_USER/vime-src/Megatron-LM" \
  "/data/$PROBE_USER/cuda-toolkit-12.9" \
  "$HOME/models/models/Qwen--Qwen3.5-9B/snapshots/master"; do
  printf '\npath=%s\n' "$directory"
  ls -ld "$directory" 2>/dev/null || true
  if [[ -d "$directory" ]]; then
    du -sh "$directory" 2>/dev/null || true
  fi
done
if [[ -d "$PROBE_PROJECT" ]]; then
  git -C "$PROBE_PROJECT" rev-parse HEAD 2>/dev/null || true
  git -C "$PROBE_PROJECT" status --short -- scripts research_agent docs 2>/dev/null || true
  for script in runtime_efficiency_pilot.py audit_runtime_efficiency.py effect_space_pilot.py; do
    if [[ -f "$PROBE_PROJECT/scripts/$script" ]]; then
      printf 'available_script=%s\n' "$script"
    else
      printf 'missing_script=%s\n' "$script"
    fi
  done
fi
for root in "$PROBE_BASE/outputs" "$PROBE_PROJECT/results"; do
  if [[ -d "$root" ]]; then
    printf '\nmodel_metadata_under=%s\n' "$root"
    find "$root/" -maxdepth 5 -type f \
      \( -name config.json -o -name model.safetensors.index.json \) -print 2>/dev/null || true
  fi
done
printf '\nprobe_completed=true\n'
