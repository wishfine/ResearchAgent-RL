"""Opt-in initialization diagnostics; preserves calls and re-raises failures.

Only constructor/init-worker methods are wrapped, not the multiprocessing
entrypoint, so spawn pickling and execution targets remain unchanged.
"""
from __future__ import annotations

import faulthandler
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

_FAULT_FILE = None  # Keep open for native fault output until process exit.


def _append(root: Path, text: str) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"worker_{os.getpid()}.log").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
    except OSError as exc:
        # Diagnostic I/O must not replace a model initialization exception.
        try:
            print(f"[RA_WORKER_DIAG] logging failed: {exc}", file=sys.stderr, flush=True)
        except OSError:
            pass


def wrap_initialization(cls, method: str, label: str, root: Path) -> None:
    original = getattr(cls, method)
    if getattr(original, "_research_agent_worker_diag", False):
        return

    @wraps(original)
    def traced(self, *args, **kwargs):
        details = {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "VLLM_WORKER_MULTIPROC_METHOD",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS",
        )}
        _append(root, f"START {label} pid={os.getpid()} " + json.dumps(details))
        try:
            result = original(self, *args, **kwargs)
        except BaseException:
            _append(root, f"FAIL {label}\n" + traceback.format_exc())
            raise
        _append(root, f"DONE {label}")
        return result

    traced._research_agent_worker_diag = True
    setattr(cls, method, traced)


def install() -> None:
    global _FAULT_FILE
    root = Path(os.environ["VIME_VLLM_WORKER_DIAG_DIR"])
    _append(root, f"START diagnostic install pid={os.getpid()}")
    try:
        if _FAULT_FILE is None:
            _FAULT_FILE = (root / f"worker_{os.getpid()}_fault.log").open("a")
            faulthandler.enable(file=_FAULT_FILE, all_threads=True)
        executor = importlib.import_module("vllm.v1.executor.multiproc_executor")
        base = importlib.import_module("vllm.v1.worker.worker_base")
        wrap_initialization(executor.WorkerProc, "__init__", "WorkerProc.__init__", root)
        wrap_initialization(base.WorkerWrapperBase, "init_worker", "WorkerWrapperBase.init_worker", root)
    except Exception:
        _append(root, "FAIL diagnostic install\n" + traceback.format_exc())
        print("[RA_WORKER_DIAG] install failed; see diagnostic files", file=sys.stderr, flush=True)
        return
    _append(root, f"DONE diagnostic install executor={executor.__file__} base={base.__file__}")
    print(f"[RA_WORKER_DIAG] installed pid={os.getpid()} dir={root}", flush=True)
