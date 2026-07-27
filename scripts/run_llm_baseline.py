#!/usr/bin/env python3
"""Run a reproducible LLM-driven ResearchEnv baseline against an OpenAI-compatible API."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research_agent.baselines.llm_actor import LLMActor
from research_agent.core.baseline_runner import aggregate_records, run_episode
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.env import ResearchEnv
from research_agent.core.env.llm_client import LLMClient
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.answer import AnswerTool
from research_agent.core.tools.cite import CiteTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.rerank import RerankTool
from research_agent.core.tools.search import SearchTool


def load_task(path: str) -> TaskSample:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return TaskSample(
        task_id=raw["task_id"],
        task_type=TaskType(raw.get("task_type", "survey_synthesis")),
        user_query=raw["user_query"],
        rubric=Rubric(),
        ground_truth_answer=raw.get("ground_truth_answer", ""),
        ground_truth_citations=raw.get("ground_truth_citations", []),
        reference_docs=raw.get("reference_docs", []),
    )


def chat_endpoint(model_url: str) -> str:
    url = model_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def models_endpoint(model_url: str) -> str:
    """Convert an OpenAI-compatible endpoint into its readiness endpoint."""
    url = model_url.rstrip("/")
    v1_position = url.find("/v1")
    if v1_position >= 0:
        return f"{url[:v1_position + 3]}/models"
    return f"{url}/v1/models"


def wait_for_model_server(model_url: str, timeout_sec: float, poll_interval_sec: float = 2.0) -> str:
    """Wait for the vLLM model-list endpoint before beginning an evaluation."""
    endpoint = models_endpoint(model_url)
    deadline = time.monotonic() + timeout_sec
    last_error = "not contacted"
    while True:
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload.get("data"), list):
                return endpoint
            last_error = f"unexpected response: {payload!r}"
        except Exception as exc:  # Readiness checks must also handle connection failures.
            last_error = str(exc)

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Model API was not ready at {endpoint} within {timeout_sec:.0f}s: {last_error}"
            )
        time.sleep(min(poll_interval_sec, max(0.0, deadline - time.monotonic())))


def make_env(corpus: CorpusStore, max_steps: int) -> ResearchEnv:
    env = ResearchEnv(corpus=corpus, max_steps=max_steps)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(RerankTool())
    env.register_tool(CiteTool())
    env.register_tool(AnswerTool())
    return env


def select_task_paths(task_paths: list[str], max_episodes: int, selection_seed: int | None) -> list[str]:
    """Choose evaluation tasks, optionally with a reproducible random sample."""
    ordered_paths = sorted(task_paths)
    if selection_seed is not None:
        random.Random(selection_seed).shuffle(ordered_paths)
    return ordered_paths[:max_episodes]


def write_records(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LLM ResearchEnv baseline and save JSONL trajectories.")
    parser.add_argument("--model_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", default="Qwen3.5-9B")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--tasks_dir", default="data/searchqa_debug/tasks")
    parser.add_argument("--corpus_dir", default="data/searchqa_debug/corpus")
    parser.add_argument("--output_dir", default="outputs/llm_baseline")
    parser.add_argument("--max_episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=None,
        help="Shuffle task files with this seed before selecting --max_episodes",
    )
    parser.add_argument(
        "--api_ready_timeout_sec",
        type=float,
        default=300.0,
        help="Maximum time to wait for /v1/models before evaluation (0 checks once)",
    )
    parser.add_argument(
        "--continue_on_actor_error",
        action="store_true",
        help="Continue after an LLM connection failure (not recommended for benchmark metrics)",
    )
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_episodes <= 0:
        parser.error("--max_episodes must be positive")
    if args.api_ready_timeout_sec < 0:
        parser.error("--api_ready_timeout_sec must be non-negative")
    if not os.path.isdir(args.tasks_dir):
        parser.error(f"tasks directory does not exist: {args.tasks_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        ready_endpoint = wait_for_model_server(args.model_url, args.api_ready_timeout_sec)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"Model API ready: {ready_endpoint}")

    corpus = CorpusStore(args.corpus_dir)
    corpus.load()
    available_task_paths = [
        os.path.join(args.tasks_dir, name)
        for name in os.listdir(args.tasks_dir)
        if name.endswith(".json")
    ]
    task_paths = select_task_paths(
        available_task_paths, args.max_episodes, selection_seed=args.selection_seed
    )
    if not task_paths:
        parser.error(f"no JSON tasks found in: {args.tasks_dir}")

    client = LLMClient(
        api_url=chat_endpoint(args.model_url), api_key=args.api_key, model=args.model_name
    )
    actor = LLMActor(client)
    records: list[dict] = []
    for index, task_path in enumerate(task_paths, start=1):
        task = load_task(task_path)
        print(f"[{index}/{len(task_paths)}] {task.task_id}: {task.user_query}")
        record = run_episode(
            make_env(corpus, args.max_steps),
            task,
            actor,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        records.append(record)
        metrics = record["eval_metrics"]
        print(
            f"  reason={record['termination_reason']} steps={metrics['average_steps']:.0f} "
            f"answer_quality={metrics['answer_quality']:.3f} citation_f1={metrics['citation_f1']:.3f}"
        )
        if record["termination_reason"] == "actor_error" and not args.continue_on_actor_error:
            partial_path = os.path.join(args.output_dir, "trajectories.partial.jsonl")
            write_records(partial_path, records)
            failure_path = os.path.join(args.output_dir, "failure.json")
            with open(failure_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "reason": "actor_error",
                        "failed_task_id": task.task_id,
                        "completed_episodes_before_failure": index - 1,
                        "requested_episodes": len(task_paths),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            print(
                "[ERROR] Evaluation stopped after an actor error; no metrics summary was written. "
                f"Partial records: {partial_path}",
                file=sys.stderr,
            )
            return 2

    trajectories_path = os.path.join(args.output_dir, "trajectories.jsonl")
    write_records(trajectories_path, records)
    summary = aggregate_records(records)
    summary_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved trajectories: {trajectories_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
