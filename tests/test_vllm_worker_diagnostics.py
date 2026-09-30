import importlib
import os
from pathlib import Path

import pytest


def test_worker_failure_is_written_and_original_exception_is_preserved(tmp_path):
    diag = importlib.import_module("scripts.vime_vllm_worker_diag")
    error = RuntimeError("underlying worker startup failure")

    class Worker:
        def __init__(self, rank):
            raise error

    diag.wrap_initialization(Worker, "__init__", "WorkerProc.__init__", tmp_path)
    with pytest.raises(RuntimeError) as caught:
        Worker(rank=1)
    assert caught.value is error
    contents = (tmp_path / f"worker_{os.getpid()}.log").read_text()
    assert "START WorkerProc.__init__" in contents
    assert "RuntimeError: underlying worker startup failure" in contents


def test_successful_initialization_and_arguments_are_unchanged(tmp_path):
    diag = importlib.import_module("scripts.vime_vllm_worker_diag")

    class Wrapper:
        def init_worker(self, value, *, flag):
            return value, flag

    diag.wrap_initialization(Wrapper, "init_worker", "init_worker", tmp_path)
    wrapped = Wrapper.init_worker
    diag.wrap_initialization(Wrapper, "init_worker", "init_worker", tmp_path)
    assert Wrapper.init_worker is wrapped
    assert Wrapper().init_worker(3, flag=True) == (3, True)
    contents = (tmp_path / f"worker_{os.getpid()}.log").read_text()
    assert "DONE init_worker" in contents


def test_log_write_failure_does_not_replace_worker_exception(tmp_path, monkeypatch):
    diag = importlib.import_module("scripts.vime_vllm_worker_diag")
    error = ValueError("real failure")

    class Worker:
        def __init__(self):
            raise error

    def unavailable(*args, **kwargs):
        raise OSError("diagnostic disk unavailable")

    monkeypatch.setattr(Path, "open", unavailable)
    diag.wrap_initialization(Worker, "__init__", "WorkerProc.__init__", tmp_path)
    with pytest.raises(ValueError) as caught:
        Worker()
    assert caught.value is error


def test_diagnostics_are_explicitly_propagated_to_ray_workers():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/run_vime_hotpotqa_smoke6.sh").read_text()
    assert '"VIME_VLLM_WORKER_DIAG": os.environ["VIME_VLLM_WORKER_DIAG"]' in script
    assert '"VIME_VLLM_WORKER_DIAG_DIR": os.environ["VIME_VLLM_WORKER_DIAG_DIR"]' in script
    hook = (root / "scripts/vime_weight_sync_trace_sitecustomize.py").read_text()
    assert 'os.environ.get("VIME_VLLM_WORKER_DIAG") == "1"' in hook
