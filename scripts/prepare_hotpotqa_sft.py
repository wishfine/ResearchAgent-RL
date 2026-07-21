#!/usr/bin/env python3
"""Build strict, grounded HotpotQA tool-use trajectories for SFT cold start.

Each teacher episode is deliberately short and follows the exact runtime
protocol: SEARCH -> READ -> CITE -> ANSWER.  Assistant turns contain one
compact JSON action and no chain-of-thought, so the generated data directly
teaches the contract enforced by :class:`ActionParser`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.env import ResearchEnv
from research_agent.core.env.rollout_collector import ConversationCollector
from research_agent.core.schema.action import Action
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.answer import AnswerTool
from research_agent.core.tools.cite import CiteTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.rerank import RerankTool
from research_agent.core.tools.search import SearchTool


def _action_text(tool: str, intent: str, params: dict[str, Any]) -> str:
    payload = {"tool": tool, "intent": intent, "params": params}
    return f"<action>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</action>"


def _load_task(path: Path) -> TaskSample:
    raw = json.loads(path.read_text(encoding="utf-8"))
    citations = raw.get("ground_truth_citations") or []
    if not raw.get("user_query") or not raw.get("ground_truth_answer") or not citations:
        raise ValueError(f"{path.name} is missing query, answer, or ground_truth_citations")
    return TaskSample(
        task_id=raw["task_id"],
        task_type=TaskType(raw.get("task_type", "survey_synthesis")),
        user_query=raw["user_query"],
        rubric=Rubric(),
        ground_truth_answer=raw["ground_truth_answer"],
        ground_truth_citations=citations,
        reference_docs=raw.get("reference_docs") or [],
    )


def _make_env(corpus: CorpusStore) -> ResearchEnv:
    env = ResearchEnv(corpus=corpus, max_steps=4)
    for tool in (SearchTool(), ReadTool(), RerankTool(), CiteTool(), AnswerTool()):
        env.register_tool(tool)
    return env


def _append_turn(collector: ConversationCollector, env: ResearchEnv, action: Action, text: str) -> tuple[bool, str]:
    collector.add_assistant_response(text)
    _, done, reason = env.step(action)
    step = env._state.trajectory[-1]
    if not step.is_valid or step.tool_result is None or not step.tool_result.success:
        raise RuntimeError(f"teacher action {action.tool} failed: {step.error_message or step.tool_result}")
    if not done:
        collector.add_observation(action.tool, True, step.tool_result.data)
    return done, reason


def build_examples(tasks_dir: str | Path, corpus_dir: str | Path) -> list[dict[str, Any]]:
    corpus = CorpusStore(str(corpus_dir))
    corpus.load()
    if not corpus.chunks:
        raise ValueError(f"No corpus chunks loaded from {corpus_dir}")

    examples: list[dict[str, Any]] = []
    for task_path in sorted(Path(tasks_dir).glob("*.json")):
        task = _load_task(task_path)
        missing = [chunk_id for chunk_id in task.ground_truth_citations if chunk_id not in corpus]
        if missing:
            raise ValueError(f"{task.task_id} citations missing from corpus: {missing}")
        env = _make_env(corpus)
        env.reset(task)
        collector = ConversationCollector()
        collector.add_user_message(task.user_query)
        citation_ids = list(task.ground_truth_citations)

        done, _ = _append_turn(
            collector,
            env,
            Action("SEARCH", "Find evidence for the question", {"query": task.user_query, "topk": 3}),
            _action_text("SEARCH", "Find evidence for the question", {"query": task.user_query, "topk": 3}),
        )
        if done:
            raise RuntimeError(f"{task.task_id} terminated after SEARCH")
        done, _ = _append_turn(
            collector,
            env,
            Action("READ", "Read the supporting evidence", {"chunk_ids": citation_ids}),
            _action_text("READ", "Read the supporting evidence", {"chunk_ids": citation_ids}),
        )
        if done:
            raise RuntimeError(f"{task.task_id} terminated after READ")
        claim = f"Evidence supports the answer: {task.ground_truth_answer}"
        done, _ = _append_turn(
            collector,
            env,
            Action("CITE", "Cite the supporting evidence", {"chunk_ids": citation_ids, "claims": [claim]}),
            _action_text("CITE", "Cite the supporting evidence", {"chunk_ids": citation_ids, "claims": [claim]}),
        )
        if done:
            raise RuntimeError(f"{task.task_id} terminated after CITE")
        done, reason = _append_turn(
            collector,
            env,
            Action("ANSWER", "Submit the grounded answer", {"answer_text": task.ground_truth_answer, "cited_chunk_ids": citation_ids}),
            _action_text("ANSWER", "Submit the grounded answer", {"answer_text": task.ground_truth_answer, "cited_chunk_ids": citation_ids}),
        )
        if not done or reason != "answer_submitted":
            raise RuntimeError(f"{task.task_id} teacher did not submit an answer: {reason}")

        examples.append({
            "format": "research-agent-sft-v1",
            "task_id": task.task_id,
            "messages": [{"role": segment.role, "content": segment.text} for segment in collector.segments],
            "metadata": {
                "ground_truth_answer": task.ground_truth_answer,
                "ground_truth_citations": citation_ids,
            },
        })
    if not examples:
        raise ValueError(f"No task JSON files found in {tasks_dir}")
    return examples


def write_examples(examples: list[dict[str, Any]], output_file: str | Path) -> dict[str, Any]:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            line = json.dumps(example, ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
    manifest = {"format": "research-agent-sft-v1", "records": len(examples), "records_sha256": digest.hexdigest()}
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict HotpotQA SFT trajectories")
    parser.add_argument("--tasks_dir", required=True)
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    examples = build_examples(args.tasks_dir, args.corpus_dir)
    manifest = write_examples(examples, args.output_file)
    print(f"Generated {manifest['records']} strict HotpotQA SFT trajectories: {args.output_file}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
