"""Opt-in initialization diagnostics; preserves calls and re-raises failures.

Observe natural module imports instead of importing vLLM during Python startup.
Only constructor/init-worker methods are wrapped, not multiprocessing targets.
"""
from __future__ import annotations

import faulthandler
from functools import wraps
import json
import os
from pathlib import Path
import sys
import traceback

_FAULT_FILE = None  # Keep open for native fault output until process exit.
_FINDER = None
_TARGETS = {
    "vllm.v1.executor.multiproc_executor": ("WorkerProc", "__init__"),
    "vllm.v1.worker.worker_base": ("WorkerWrapperBase", "init_worker"),
}


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


def _enable_fault_capture(root: Path) -> None:
    global _FAULT_FILE
    try:
        if _FAULT_FILE is None:
            root.mkdir(parents=True, exist_ok=True)
            _FAULT_FILE = (root / f"worker_{os.getpid()}_fault.log").open("a")
            faulthandler.enable(file=_FAULT_FILE, all_threads=True)
    except Exception:
        _append(root, "FAIL fault capture setup\n" + traceback.format_exc())


def _instrument(module, root: Path) -> None:
    try:
        cls_name, method = _TARGETS[module.__name__]
        wrap_initialization(getattr(module, cls_name), method, f"{cls_name}.{method}", root)
        _append(root, f"HOOKED {module.__name__} file={getattr(module, '__file__', None)}")
    except Exception:
        # Diagnostic setup must not turn a successful application import into a failure.
        _append(root, "FAIL diagnostic wrapper setup\n" + traceback.format_exc())


class _DiagnosticLoader:
    def __init__(self, loader, root: Path):
        self.loader = loader
        self.root = root

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        _enable_fault_capture(self.root)
        _append(self.root, f"IMPORT {module.__name__} pid={os.getpid()}")
        try:
            self.loader.exec_module(module)
        except BaseException:
            _append(self.root, f"FAIL import {module.__name__}\n" + traceback.format_exc())
            raise
        _instrument(module, self.root)


class _DiagnosticFinder:
    def __init__(self, root: Path):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        # Python has already tried any finders preceding us. Only delegate to
        # the remaining suffix (other hooks may be inserted ahead of us later).
        finders = tuple(sys.meta_path)
        for finder in finders[finders.index(self) + 1:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _DiagnosticLoader(spec.loader, self.root)
                return spec
        return None


def install() -> None:
    global _FINDER
    if _FINDER is not None:
        return
    root = Path(os.environ["VIME_VLLM_WORKER_DIAG_DIR"])
    _FINDER = _DiagnosticFinder(root)
    sys.meta_path.insert(0, _FINDER)
    # Handle a late/manual install without importing any missing modules.
    for name in _TARGETS:
        module = sys.modules.get(name)
        if module is not None:
            _enable_fault_capture(root)
            _instrument(module, root)
    _append(root, f"DONE deferred diagnostic install pid={os.getpid()} (no eager vLLM imports)")
