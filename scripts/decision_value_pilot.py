#!/usr/bin/env python3
"""Diagnose whether answer-entropy reduction tracks useful SEARCH decisions.

This is an offline, frozen-model diagnostic. Gold answers are used only after
generation to score saved records; they are never included in model prompts.
The observed accuracy change is *not* an estimate of expected value of
information and must not be used as an online reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.adapters.slime.custom_reward import compute_metrics, normalize_answer
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.llm_client import LLMClient
from scripts.run_llm_baseline import chat_endpoint, wait_for_model_server


def chat_prompt(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n")


def parse_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def parse_queries(text: str, question: str, max_queries: int) -> list[str]:
    proposed = parse_json_object(text).get("queries", [])
    queries = [question]
    if isinstance(proposed, list):
        queries.extend(value.strip() for value in proposed if isinstance(value, str))
    distinct = []
    seen = set()
    for query in queries:
        key = " ".join(query.casefold().split())
        if key and key not in seen:
            distinct.append(query)
            seen.add(key)
        if len(distinct) >= max_queries:
            break
    return distinct


def parse_answer(text: str) -> str:
    value = parse_json_object(text).get("answer")
    if isinstance(value, str):
        return value.strip()
    # Keep failed-format outputs visible as failures, not secretly repaired.
    return ""


def answer_distribution(samples: list[str]) -> tuple[str, float]:
    if not samples:
        raise ValueError("At least one answer sample is required")
    counts = Counter(normalize_answer(sample) for sample in samples)
    mode = min(counts, key=lambda key: (-counts[key], key))
    entropy = -sum((count / len(samples)) * math.log(count / len(samples))
                   for count in counts.values())
    return mode, entropy


def answer_quality(answer: str, task: dict) -> float:
    references = [task["ground_truth_answer"],
                  *(task.get("ground_truth_answer_aliases") or [])]
    return max(compute_metrics(answer, reference, [], [])["answer_quality"]
               for reference in references)


def analyze(records: list[dict]) -> dict:
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    per_task = []
    answer_outputs = 0
    parsed_answers = 0
    for task_id, group in sorted(by_task.items()):
        if len({row["query"] for row in group}) != len(group):
            raise ValueError(f"Duplicate query in task {task_id}")
        prior_samples = group[0]["prior_samples"]
        if any(row["prior_samples"] != prior_samples for row in group):
            raise ValueError(f"Inconsistent prior samples in task {task_id}")
        task = group[0]["task"]
        all_samples = [*prior_samples, *(sample for row in group
                                         for sample in row["posterior_samples"])]
        answer_outputs += len(all_samples)
        parsed_answers += sum(bool(normalize_answer(sample)) for sample in all_samples)
        if any(not normalize_answer(sample) for sample in all_samples):
            per_task.append({"task_id": task_id, "n_queries": len(group),
                             "usable": False, "reason": "unparseable_answer_sample"})
            continue
        prior_mode, prior_entropy = answer_distribution(prior_samples)
        prior_quality = answer_quality(prior_mode, task)
        options = []
        for row in group:
            if row["task"] != task:
                raise ValueError(f"Inconsistent task metadata in {task_id}")
            mode, entropy = answer_distribution(row["posterior_samples"])
            quality = answer_quality(mode, task)
            options.append({
                "query": row["query"], "entropy_reduction": prior_entropy - entropy,
                "observed_quality_change": quality - prior_quality,
                "posterior_quality": quality, "posterior_entropy": entropy,
                "retrieved_chunk_ids": row["retrieved_chunk_ids"],
            })
        entropy_best = max(options, key=lambda x: (x["entropy_reduction"], x["query"]))
        quality_best = max(options, key=lambda x: (x["posterior_quality"], x["query"]))
        entropy_spread = (max(option["entropy_reduction"] for option in options)
                          - min(option["entropy_reduction"] for option in options))
        per_task.append({
            "task_id": task_id, "n_queries": len(options), "usable": True,
            "entropy_rankable": len(options) >= 2 and entropy_spread > 1e-6,
            "prior_quality": prior_quality, "prior_entropy": prior_entropy,
            "entropy_selected_quality": entropy_best["posterior_quality"],
            "best_observed_quality": quality_best["posterior_quality"],
            "selection_regret_diagnostic": (
                quality_best["posterior_quality"] - entropy_best["posterior_quality"]
            ),
            "entropy_selected_query": entropy_best["query"],
            "quality_best_query_oracle": quality_best["query"],
            "options": options,
        })
    comparable = [row for row in per_task if row["usable"] and row["n_queries"] >= 2]
    rankable = [row for row in comparable if row["entropy_rankable"]]
    return {
        "protocol": "frozen-model-search-decision-diagnostic-v1",
        "tasks": len(per_task), "comparable_tasks": len(comparable),
        "entropy_rankable_tasks": len(rankable),
        "answer_parse_success_rate": (
            parsed_answers / answer_outputs if answer_outputs else None
        ),
        "entropy_ranking_regret_rate": (
            sum(row["selection_regret_diagnostic"] > 0 for row in rankable) / len(rankable)
            if rankable else None
        ),
        "mean_entropy_ranking_regret": (
            sum(row["selection_regret_diagnostic"] for row in rankable) / len(rankable)
            if rankable else None
        ),
        "per_task": per_task,
        "limitations": [
            "Gold answers are used only for retrospective evaluation, never in prompts.",
            "The best-quality query is an oracle diagnostic, not a deployable policy.",
            "One retrieved result per query does not identify expected value of information.",
            "Answer-sample entropy depends on model sampling and is not calibrated uncertainty.",
            "This pilot does not train or evaluate an RL policy.",
        ],
    }


def answer_samples(client: LLMClient, question: str, evidence: list[dict],
                   count: int, temperature: float) -> tuple[list[str], int]:
    context = "\n".join(
        f"[{item['chunk_id']}] {item['title']}: {item['content'][:300]}"
        for item in evidence
    ) or "No documents provided."
    prompt = chat_prompt(
        "Answer the question. Use the provided documents when available; when none "
        "are provided, use your existing knowledge. If uncertain, answer UNKNOWN. "
        "Return exactly one JSON object with an "
        "'answer' string field; no explanation.",
        f"Question: {question}\nDocuments:\n{context}",
    )
    outputs = []
    total_tokens = 0
    for _ in range(count):
        response = client.generate_response(prompt, max_tokens=256,
                                            temperature=temperature, top_p=0.95)
        outputs.append(parse_answer(response.content))
        total_tokens += response.prompt_tokens + response.completion_tokens
    return outputs, total_tokens


def collect_task(task: dict, corpus: CorpusStore, client: LLMClient, *,
                 answer_count: int, max_queries: int, topk: int,
                 temperature: float) -> list[dict]:
    question = task["user_query"]
    query_prompt = chat_prompt(
        "Produce distinct corpus search queries for finding evidence to answer "
        "the question. Do not answer it. Return JSON only: "
        '{"queries": ["...", "..."]}.',
        question,
    )
    proposal = client.generate_response(query_prompt, max_tokens=320, temperature=0.7)
    queries = parse_queries(proposal.content, question, max_queries)
    prior, prior_tokens = answer_samples(client, question, [], answer_count, temperature)
    rows = []
    for query in queries:
        candidates = corpus.search(query, topk=topk,
                                   doc_ids=task.get("reference_docs") or None)
        evidence = []
        for candidate in candidates:
            chunk = corpus.get_chunk(candidate.chunk_id)
            if chunk is not None:
                evidence.append({"chunk_id": chunk.chunk_id, "title": chunk.title,
                                 "content": chunk.content})
        posterior, tokens = answer_samples(client, question, evidence,
                                           answer_count, temperature)
        rows.append({
            "task_id": task["task_id"], "task": {
                "ground_truth_answer": task["ground_truth_answer"],
                "ground_truth_answer_aliases": task.get("ground_truth_answer_aliases") or [],
            },
            "query": query, "prior_samples": prior, "posterior_samples": posterior,
            "retrieved_chunk_ids": [item["chunk_id"] for item in evidence],
            "model_tokens": tokens + (prior_tokens + proposal.prompt_tokens
                                       + proposal.completion_tokens if query == queries[0] else 0),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks_dir", type=Path)
    parser.add_argument("--corpus_dir", type=Path)
    parser.add_argument("--model_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", default="Qwen3.5-9B")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--answer_samples", type=int, default=4)
    parser.add_argument("--max_queries", type=int, default=3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()
    if args.max_tasks <= 0 or args.answer_samples <= 0 or args.max_queries < 2 or args.topk <= 0:
        parser.error("max_tasks, answer_samples, topk must be positive; max_queries >= 2")
    if not 0 < args.temperature <= 2:
        parser.error("temperature must be in (0, 2]")
    records_path = args.output_dir / "records.jsonl"
    config_path = args.output_dir / "config.json"
    if args.analyze_only:
        if not records_path.is_file():
            parser.error(f"Missing records: {records_path}")
    else:
        if not args.tasks_dir or not args.corpus_dir:
            parser.error("--tasks_dir and --corpus_dir are required for collection")
        paths = sorted(args.tasks_dir.glob("*.json"))
        random.Random(args.selection_seed).shuffle(paths)
        paths = paths[:args.max_tasks]
        if not paths:
            parser.error(f"No task files in {args.tasks_dir}")
        selected_hash = hashlib.sha256()
        for path in paths:
            selected_hash.update(path.name.encode())
            selected_hash.update(path.read_bytes())
        config = {
            "tasks_dir": str(args.tasks_dir.resolve()),
            "corpus_dir": str(args.corpus_dir.resolve()), "model_url": args.model_url,
            "model_name": args.model_name, "max_tasks": args.max_tasks,
            "selection_seed": args.selection_seed, "answer_samples": args.answer_samples,
            "max_queries": args.max_queries, "topk": args.topk,
            "temperature": args.temperature, "selected_tasks_sha256": selected_hash.hexdigest(),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if not config_path.is_file() or json.loads(config_path.read_text()) != config:
                parser.error("Cannot resume: collection config does not match")
        elif config_path.exists() or records_path.exists():
            parser.error("Output exists; use --resume or a fresh output directory")
        else:
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        task_records_dir = args.output_dir / "task_records"
        task_records_dir.mkdir(exist_ok=True)
        wait_for_model_server(args.model_url, timeout_sec=30)
        corpus = CorpusStore(str(args.corpus_dir))
        corpus.load()
        client = LLMClient(api_url=chat_endpoint(args.model_url), model=args.model_name)
        try:
            for index, path in enumerate(paths, 1):
                task = json.loads(path.read_text(encoding="utf-8"))
                task_id = task["task_id"]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
                    raise ValueError(f"Unsafe task ID: {task_id!r}")
                record_path = task_records_dir / f"{task_id}.json"
                if record_path.is_file():
                    continue
                rows = collect_task(task, corpus, client,
                                    answer_count=args.answer_samples,
                                    max_queries=args.max_queries, topk=args.topk,
                                    temperature=args.temperature)
                temporary_path = record_path.with_suffix(".json.tmp")
                temporary_path.write_text(json.dumps(rows, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
                temporary_path.replace(record_path)
                print(f"[{index}/{len(paths)}] {task_id}: {len(rows)} queries",
                      flush=True)
        finally:
            corpus.close()
        all_rows = []
        for path in paths:
            task = json.loads(path.read_text(encoding="utf-8"))
            record_path = task_records_dir / f"{task['task_id']}.json"
            all_rows.extend(json.loads(record_path.read_text(encoding="utf-8")))
        temporary_records = records_path.with_suffix(".jsonl.tmp")
        temporary_records.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
            encoding="utf-8",
        )
        temporary_records.replace(records_path)
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    summary = analyze(records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_task"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
