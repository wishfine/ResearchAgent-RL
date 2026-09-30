"""Diagnose silent init_worker exits without training, init_device, or load_model.

Submit a CPU-only job to an existing Ray dashboard using a failed run's saved
runtime environment. A separate child observes GPU6/7 visibility, matching the
failed rollout, but only constructs config and calls WorkerWrapperBase.init_worker.
No Ray cluster is created/stopped. Import/constructor code may still query CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys


CHILD_CODE = r'''
import linecache
import multiprocessing
import sys

print("PROBE_CONFIG_START", flush=True)
from vllm.engine.arg_utils import EngineArgs
config = EngineArgs(
    model=sys.argv[1], trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=2, gpu_memory_utilization=0.7, seed=1234,
    logprobs_mode="processed_logprobs", weight_transfer_config={"backend": "nccl"},
).create_engine_config()
print("PROBE_CONFIG_DONE", flush=True)

def trace(frame, event, arg):
    filename = frame.f_code.co_filename
    if filename.endswith(("/worker_base.py", "/gpu_worker.py")):
        if event == "line":
            source = linecache.getline(filename, frame.f_lineno).strip()
            print(f"PROBE_LINE {filename}:{frame.f_lineno} {source}", flush=True)
        return trace
    return None

sys.settrace(trace)
from vllm.v1.worker.worker_base import WorkerWrapperBase
lock = multiprocessing.get_context("spawn").Lock()
wrapper = WorkerWrapperBase(rpc_rank=0, global_rank=0)
print("PROBE_INIT_WORKER_CALL", flush=True)
wrapper.init_worker([dict(
    vllm_config=config, local_rank=0, rank=0,
    distributed_init_method="tcp://127.0.0.1:1",
    is_driver_worker=True, shared_worker_lock=lock,
), {}])
sys.settrace(None)
print("INIT_WORKER_ONLY_OK", flush=True)
'''


def read_original_launch(run: Path) -> tuple[dict, str]:
    for line in (run / "driver.log").read_text().splitlines():
        if not line.startswith("+ ") or "--runtime-env-json=" not in line:
            continue
        tokens = shlex.split(line[2:])
        if not any(tokens[i:i + 2] == ["job", "submit"] for i in range(len(tokens))):
            continue
        runtime_arg = next(arg for arg in tokens if arg.startswith("--runtime-env-json="))
        runtime = json.loads(runtime_arg.split("=", 1)[1])
        model = tokens[tokens.index("--hf-checkpoint") + 1]
        return runtime, model
    raise ValueError("No original job submit runtime-env found in driver.log")


def run_probe_child(model: str, gpu_mask: str, timeout: float = 60) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_mask, HIP_VISIBLE_DEVICES=gpu_mask)
    # Match Vime's _build_subprocess_env without importing Vime/Ray in the child.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")
    print(f"PROBE_CHILD_START gpu_mask={gpu_mask} timeout={timeout}", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", CHILD_CODE, model], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, end="", flush=True)
        print("PROBE_RESULT=timeout (child killed by this diagnostic timeout)", flush=True)
        return 1
    print(result.stdout, end="", flush=True)
    print(f"PROBE_CHILD_EXIT_CODE={result.returncode}", flush=True)
    if result.returncode == 0 and "INIT_WORKER_ONLY_OK" in result.stdout.splitlines():
        print("PROBE_RESULT=completed", flush=True)
        return 0
    if result.returncode < 0:
        reason = "signal:" + signal.Signals(-result.returncode).name
    else:
        reason = "early_exit"
    print(f"PROBE_RESULT={reason}", flush=True)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--address", default="http://127.0.0.1:8301")
    parser.add_argument("--gpu-mask", default="6,7")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not re.fullmatch(r"\d+,\d+", args.gpu_mask) or len(set(args.gpu_mask.split(","))) != 2:
        parser.error("gpu-mask must contain two distinct numeric device IDs")
    if not 0 < args.timeout <= 120:
        parser.error("timeout must be in (0, 120] seconds")
    runtime, model = read_original_launch(args.run_dir)
    if not (Path(model) / "config.json").is_file():
        parser.error("original HF checkpoint config.json is missing")
    busy = subprocess.check_output([
        "nvidia-smi", "-i", args.gpu_mask, "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True).strip()
    if busy:
        parser.error(f"GPU {args.gpu_mask} busy; no job submitted:\n{busy}")
    out = args.run_dir / ("init_worker_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    runtime["env_vars"]["VIME_VLLM_WORKER_DIAG_DIR"] = str(out / "worker_debug")
    parent_code = (
        "import sys; from scripts.probe_vllm_init_worker import run_probe_child; "
        "sys.exit(run_probe_child(sys.argv[1], sys.argv[2], float(sys.argv[3])))"
    )
    cmd = [
        sys.executable, "-u", "-c", "from ray.scripts.scripts import main; main()",
        "job", "submit", "--address=" + args.address,
        "--runtime-env-json=" + json.dumps(runtime), "--",
        sys.executable, "-u", "-c", parent_code, model, args.gpu_mask, str(args.timeout),
    ]
    print(f"PROBE_DIR={out}", flush=True)
    print("No training, no init_device/load_model; existing Ray only.", flush=True)
    with (out / "probe.log").open("w") as log:
        proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=""),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
