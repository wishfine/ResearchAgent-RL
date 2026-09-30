from argparse import Namespace
from pathlib import Path
import os
import subprocess

from research_agent.adapters.vime import custom_rollout
from research_agent.core.env.rollout_collector import ADAPTIVE_SYSTEM_PROMPT
from research_agent.core.schema.task import TaskSample, TaskType, Rubric


def test_vime_full_read_opt_in(monkeypatch):
    monkeypatch.setenv("RESEARCH_AGENT_CORPUS_DIR", "data/searchqa_debug/corpus")
    monkeypatch.setenv("RESEARCH_AGENT_READ_MODE", "full")
    monkeypatch.setenv("RESEARCH_AGENT_MAX_STEPS", "15")
    env = custom_rollout._make_env(Namespace())
    assert env._tools["READ"].mode == "full"
    assert env.max_steps == 15


def test_vime_prompt_mode_opt_in(monkeypatch):
    task = TaskSample("t", TaskType.SURVEY_SYNTHESIS, "q", Rubric(), retrieval_scope="split_corpus")
    monkeypatch.setenv("RESEARCH_AGENT_PROMPT_MODE", "adaptive")
    assert custom_rollout._make_collector(task).segments[0].text == ADAPTIVE_SYSTEM_PROMPT


def test_fast_grpo_entry_config_without_gpu_side_effects():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", "scripts/run_vime_musique_grpo.sh", "--print-config"],
                            cwd=root, capture_output=True, text=True,
                            env={**os.environ, "USER": "tester"})
    assert result.returncode == 0, result.stderr
    assert "ACTOR_GPUS=4 ROLLOUT_GPUS=2" in result.stdout
    assert "N_SAMPLES_PER_PROMPT=8" in result.stdout
    assert "READ_MODE=full PROMPT_MODE=adaptive" in result.stdout
    assert "ATTENTION_BACKEND=auto" in result.stdout
    assert "NUM_ROLLOUT=200" in result.stdout
    assert "CUDA_HOME=/local_data/tester/cuda-toolkit-12.9\n" in result.stdout
