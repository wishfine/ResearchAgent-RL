import importlib
import json
import shlex
import sys

import pytest


def test_probe_available_without_importing_gpu_dependencies():
    assert importlib.util.find_spec("scripts.probe_vllm_init_worker") is not None


@pytest.fixture
def probe():
    assert importlib.util.find_spec("scripts.probe_vllm_init_worker") is not None, (
        "init-worker-only diagnostic script is missing"
    )
    return importlib.import_module("scripts.probe_vllm_init_worker")


def test_extract_original_runtime_and_model_without_executing_command(probe, tmp_path):
    runtime = {"env_vars": {"PYTHONPATH": "/original/site:/project", "UNCHANGED": "a b"}}
    args = ["/env/bin/ray", "job", "submit", "--address=http://127.0.0.1:8301",
            "--runtime-env-json=" + json.dumps(runtime), "--", "/env/bin/python",
            "/vime/train_async.py", "--hf-checkpoint", "/model with spaces", "--num-rollout", "2"]
    (tmp_path / "driver.log").write_text("+ " + shlex.join(args) + "\n")
    actual, model = probe.read_original_launch(tmp_path)
    assert actual == runtime
    assert model == "/model with spaces"


def test_missing_original_launch_fails_before_submission(probe, tmp_path):
    (tmp_path / "driver.log").write_text("not a launch command\n")
    with pytest.raises(ValueError, match="runtime-env"):
        probe.read_original_launch(tmp_path)


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    for package in ("vllm", "vllm/engine", "vllm/v1", "vllm/v1/worker"):
        directory = tmp_path / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("")
    (tmp_path / "vllm/engine/arg_utils.py").write_text(
        "class EngineArgs:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def create_engine_config(self): return object()\n"
    )
    worker = tmp_path / "vllm/v1/worker/worker_base.py"
    worker.write_text(
        "class WorkerWrapperBase:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def init_worker(self, all_kwargs):\n"
        "        assert all_kwargs[0]['rank'] == 0\n"
        "    def init_device(self): raise AssertionError('must not init device')\n"
        "    def load_model(self): raise AssertionError('must not load model')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return worker


def test_child_only_initializes_worker_without_loading_model(probe, fake_vllm, capsys):
    assert probe.run_probe_child("/fake/model", "6,7", 10) == 0
    output = capsys.readouterr().out
    assert "INIT_WORKER_ONLY_OK" in output
    assert "PROBE_CHILD_EXIT_CODE=0" in output
    assert "PROBE_RESULT=completed" in output


@pytest.mark.parametrize("code", [0, 23])
def test_silent_exit_is_reported_even_when_exit_code_is_zero(probe, fake_vllm, capsys, code):
    fake_vllm.write_text(
        "import os\nclass WorkerWrapperBase:\n"
        "    def __init__(self, **kwargs): pass\n"
        f"    def init_worker(self, all_kwargs): os._exit({code})\n"
    )
    assert probe.run_probe_child("/fake/model", "6,7", 10) == 1
    output = capsys.readouterr().out
    assert f"PROBE_CHILD_EXIT_CODE={code}" in output
    assert "PROBE_RESULT=early_exit" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal return codes")
def test_signal_exit_is_reported_separately(probe, fake_vllm, capsys):
    fake_vllm.write_text(
        "import os, signal\nclass WorkerWrapperBase:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def init_worker(self, all_kwargs): os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    assert probe.run_probe_child("/fake/model", "6,7", 10) == 1
    output = capsys.readouterr().out
    assert "PROBE_CHILD_EXIT_CODE=-9" in output
    assert "PROBE_RESULT=signal:SIGKILL" in output


def test_timeout_is_not_misreported_as_an_unexplained_signal(probe, fake_vllm, capsys):
    fake_vllm.write_text(
        "import time\nclass WorkerWrapperBase:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def init_worker(self, all_kwargs): time.sleep(10)\n"
    )
    assert probe.run_probe_child("/fake/model", "6,7", 0.5) == 1
    assert "PROBE_RESULT=timeout" in capsys.readouterr().out
