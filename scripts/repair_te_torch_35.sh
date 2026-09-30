#!/usr/bin/env bash
# Build only TE PyTorch2.10 binding against35's libc. No system-glibc changes.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-/local_data/$USER/research-agent-rl-data}"
ENV="${TRAIN_ENV:-/local_data/$USER/conda_envs/research-agent-runtime}"
PY="$ENV/bin/python"
SITE="$ENV/lib/python3.11/site-packages"
export CUDA_HOME="${CUDA_HOME:-/local_data/$USER/cuda-toolkit-12.9}"
export CUDA_PATH="$CUDA_HOME"
ARCHIVE="${TE_TORCH_ARCHIVE:-/local_data/$USER/vime-src/wheels/transformer_engine_torch-2.10.0.tar.gz}"
RUN_DIR="${RUN_DIR:-$BASE/runtime_efficiency/te_torch_native35_$(date +%Y%m%d_%H%M%S)}"
test -x "$PY"
test -x "$CUDA_HOME/bin/nvcc"
test -f "$CUDA_HOME/include/cuda_runtime.h"
test -f "$ARCHIVE"
test -x /usr/bin/g++
test -x /usr/bin/gcc
mkdir -p "$RUN_DIR/wheelhouse" "$RUN_DIR/tmp" "$RUN_DIR/old_so_backup"
if find "$RUN_DIR/wheelhouse" -maxdepth 1 -name '*.whl' -print -quit | grep -q .; then
  echo "[ERROR] Existing wheel output; choose a fresh RUN_DIR." >&2
  exit 2
fi
NVIDIA_LIBS="$(find "$SITE/nvidia" -type d -name lib -print | paste -sd: -)"
export PATH="$ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT:/local_data/$USER/vime-src/Megatron-LM:/local_data/$USER/vime-src/vime${PYTHONPATH:+:$PYTHONPATH}"
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cublas/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/cusparse/include:$SITE/nvidia/cusolver/include:$SITE/nvidia/nccl/include:$SITE/nvidia/cuda_runtime/include${CPATH:+:$CPATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ LDSHARED="/usr/bin/g++ -shared"
export TMPDIR="$RUN_DIR/tmp" PIP_CACHE_DIR="$BASE/pip_cache"
export NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_FRAMEWORK=pytorch
export MAX_JOBS="${MAX_JOBS:-8}"
export NVTE_BUILD_MAX_JOBS="$MAX_JOBS"
echo "RUN_DIR=$RUN_DIR CUDA_HOME=$CUDA_HOME"
getconf GNU_LIBC_VERSION
"$CUDA_HOME/bin/nvcc" --version
"$PY" - "$ARCHIVE" <<'PY'
import hashlib
import importlib.metadata as md
from pathlib import Path
import sys
import torch
import pybind11
import ninja
archive = Path(sys.argv[1])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == "71faff8e3def742553ad74b4e32d2d12e91be9acfb13d1699c89e1e18dd4ecd6", "Unexpected TE source archive; stop before building"
for package in ("transformer-engine", "transformer-engine-cu12", "transformer-engine-torch"):
    assert md.version(package) == "2.10.0", (package, md.version(package))
assert torch.version.cuda == "12.9", torch.version.cuda
print("torch:", torch.__version__, "CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("pybind11:", pybind11.__version__)
PY

# Never install a downloaded/cached prebuilt binary as the supposed native fix.
CUDA_VISIBLE_DEVICES="" "$PY" -m pip wheel -v --no-index --no-deps \
  --no-build-isolation --no-cache-dir --wheel-dir "$RUN_DIR/wheelhouse" "$ARCHIVE"
shopt -s nullglob
WHEELS=("$RUN_DIR"/wheelhouse/transformer_engine_torch-2.10.0-*.whl)
[[ ${#WHEELS[@]} -eq 1 ]] || { echo "Expected exactly one built wheel" >&2; exit 2; }
WHEEL="${WHEELS[0]}"
"$PY" - "$WHEEL" "$RUN_DIR" <<'PY'
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
wheel, root = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(wheel) as archive:
    names = [name for name in archive.namelist() if name.endswith(".so") and "transformer_engine_torch" in name]
    assert len(names) == 1, names
    binary = root / "new_transformer_engine_torch.so"
    binary.write_bytes(archive.read(names[0]))
output = subprocess.check_output(["readelf", "--version-info", str(binary)], text=True)
required = {tuple(map(int, version.split("."))) for version in re.findall(r"GLIBC_(\d+\.\d+(?:\.\d+)?)", output)}
native = tuple(map(int, os.confstr("CS_GNU_LIBC_VERSION").split()[-1].split(".")))
assert required and max(required) <= native, ("Wheel requires newer glibc", required, native)
print("New binding GLIBC max:", max(required), "native:", native)
PY
OLD_BINARIES=("$SITE"/transformer_engine/wheel_lib/transformer_engine_torch*.so)
[[ ${#OLD_BINARIES[@]} -ge 1 ]] || { echo "No existing TE binding found for backup; stopped before install" >&2; exit 2; }
for binary in "${OLD_BINARIES[@]}"; do
  cp -a "$binary" "$RUN_DIR/old_so_backup/"
done
sha256sum "$WHEEL"
"$PY" -m pip install --no-index --no-deps --force-reinstall "$WHEEL"
"$PY" - <<'PY'
import torch
import transformer_engine.pytorch
import megatron.core
import mbridge
print("NATIVE_TE_TORCH_IMPORT_OK", torch.__version__)
PY
echo "Re-run scripts/probe_vime_training_35.sh; GPU training is not yet verified."
