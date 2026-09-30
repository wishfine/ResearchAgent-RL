import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def diagnostic_process(tmp_path):
    # Small real modules exercise Python's import/spawn machinery without GPUs.
    for package in ("vllm", "vllm/v1", "vllm/v1/executor", "vllm/v1/worker"):
        directory = tmp_path / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("")
    (tmp_path / "vllm/v1/executor/multiproc_executor.py").write_text(
        "class WorkerProc:\n"
        "    def __init__(self, rank):\n"
        "        raise ValueError(f'rank failed: {rank}')\n"
        "def spawn_target():\n"
        "    WorkerProc(7)\n"
    )
    (tmp_path / "vllm/v1/worker/worker_base.py").write_text(
        "class WorkerWrapperBase:\n"
        "    def init_worker(self, value, *, flag):\n"
        "        return value, flag\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(root))),
               VIME_VLLM_WORKER_DIAG_DIR=str(tmp_path / "logs"),
               PYTHONDONTWRITEBYTECODE="1")

    def run(source):
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    return run, tmp_path


def test_install_does_not_import_vllm_or_enable_fault_capture(diagnostic_process):
    run, _ = diagnostic_process
    run("""
        import sys
        from pathlib import Path
        import os
        from scripts.vime_vllm_worker_diag import install
        install()
        install()
        assert not any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
        assert not list(Path(os.environ['VIME_VLLM_WORKER_DIAG_DIR']).glob('*_fault.log'))
    """)


def test_natural_import_installs_wrappers_without_importing_other_target(diagnostic_process):
    run, tmp_path = diagnostic_process
    run("""
        import sys
        from scripts.vime_vllm_worker_diag import install
        install()
        from vllm.v1.executor.multiproc_executor import WorkerProc
        assert 'vllm.v1.worker.worker_base' not in sys.modules
        try:
            WorkerProc(3)
        except ValueError as exc:
            assert str(exc) == 'rank failed: 3'
        else:
            raise AssertionError('original exception was swallowed')
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        assert WorkerWrapperBase().init_worker(9, flag=True) == (9, True)
    """)
    logs = "\n".join(p.read_text() for p in (tmp_path / "logs").glob("worker_*.log"))
    assert "FAIL WorkerProc.__init__" in logs
    assert "ValueError: rank failed: 3" in logs
    assert "DONE WorkerWrapperBase.init_worker" in logs


def test_spawn_target_is_picklable_and_child_failure_is_captured(diagnostic_process):
    run, tmp_path = diagnostic_process
    (tmp_path / "sitecustomize.py").write_text(
        "from scripts.vime_vllm_worker_diag import install\ninstall()\n"
    )
    run("""
        import multiprocessing
        from vllm.v1.executor.multiproc_executor import spawn_target
        if __name__ == '__main__':
            child = multiprocessing.get_context('spawn').Process(target=spawn_target)
            child.start()
            child.join(15)
            if child.is_alive():
                child.terminate()
                child.join()
                raise AssertionError('spawn child timed out')
            assert child.exitcode == 1, child.exitcode
    """)
    logs = "\n".join(p.read_text() for p in (tmp_path / "logs").glob("worker_*.log"))
    assert "FAIL WorkerProc.__init__" in logs
    assert "ValueError: rank failed: 7" in logs


def test_late_install_wraps_loaded_target_without_importing_missing_one(diagnostic_process):
    run, _ = diagnostic_process
    run("""
        import sys
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from scripts.vime_vllm_worker_diag import install
        install()
        assert 'vllm.v1.executor.multiproc_executor' not in sys.modules
        assert WorkerWrapperBase.init_worker._research_agent_worker_diag
        assert WorkerWrapperBase().init_worker(2, flag=False) == (2, False)
    """)


def test_original_module_import_failure_is_preserved(diagnostic_process):
    run, tmp_path = diagnostic_process
    (tmp_path / "sentinel.py").write_text("error = RuntimeError('original import failure')\n")
    (tmp_path / "vllm/v1/executor/multiproc_executor.py").write_text(
        "from sentinel import error\nraise error\n"
    )
    run("""
        from scripts.vime_vllm_worker_diag import install
        from sentinel import error
        install()
        try:
            import vllm.v1.executor.multiproc_executor
        except RuntimeError as exc:
            assert exc is error
        else:
            raise AssertionError('original import exception was swallowed')
    """)
    logs = "\n".join(p.read_text() for p in (tmp_path / "logs").glob("worker_*.log"))
    assert "FAIL import vllm.v1.executor.multiproc_executor" in logs
    assert "RuntimeError: original import failure" in logs


def test_finder_added_ahead_of_observer_is_not_called_twice(diagnostic_process):
    run, _ = diagnostic_process
    run("""
        import sys
        from scripts.vime_vllm_worker_diag import install
        install()
        class AheadFinder:
            calls = 0
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'vllm.v1.executor.multiproc_executor':
                    self.calls += 1
                    if self.calls > 1:
                        raise RuntimeError('finder called twice')
                return None
        ahead = AheadFinder()
        sys.meta_path.insert(0, ahead)
        from vllm.v1.executor.multiproc_executor import WorkerProc
        assert ahead.calls == 1
        assert WorkerProc.__init__._research_agent_worker_diag
    """)


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
