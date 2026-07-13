#!/usr/bin/env python3
"""Run a reproducible LLM-driven ResearchEnv baseline against an OpenAI-compatible API."""
from __future__ import annotations

import argparse
import json
import os
import sys

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


def make_env(corpus: CorpusStore, max_steps: int) -> ResearchEnv:
    env = ResearchEnv(corpus=corpus, max_steps=max_steps)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(RerankTool())
    env.register_tool(CiteTool())
    env.register_tool(AnswerTool())
    return env


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
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_episodes <= 0:
        parser.error("--max_episodes must be positive")
    if not os.path.isdir(args.tasks_dir):
        parser.error(f"tasks directory does not exist: {args.tasks_dir}")

    corpus = CorpusStore(args.corpus_dir)
    corpus.load()
    task_paths = sorted(
        os.path.join(args.tasks_dir, name)
        for name in os.listdir(args.tasks_dir)
        if name.endswith(".json")
    )[: args.max_episodes]
    if not task_paths:
        parser.error(f"no JSON tasks found in: {args.tasks_dir}")

    client = LLMClient(
        api_url=chat_endpoint(args.model_url), api_key=args.api_key, model=args.model_name
    )
    actor = LLMActor(client)
    os.makedirs(args.output_dir, exist_ok=True)
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

    trajectories_path = os.path.join(args.output_dir, "trajectories.jsonl")
    with open(trajectories_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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
