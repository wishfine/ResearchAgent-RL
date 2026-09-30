#!/usr/bin/env python3
"""Measure whether sampled SEARCH queries have redundant *tool effects*.

Sampling and retrieval analysis are separated. An existing samples.jsonl can be
reanalyzed without spending model calls. This is a feasibility diagnostic, not
an RL training run or a claim about downstream value equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.baselines.llm_actor import LLMActor
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.effect_space import SearchProposal, effect_key, group_by_effect, jaccard
from research_agent.core.env.llm_client import LLMClient
from research_agent.core.env.rollout_collector import ConversationCollector
from scripts.run_llm_baseline import chat_endpoint, wait_for_model_server


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_first_search(
    tasks_dir: Path, model_url: str, model_name: str, *, max_tasks: int,
    samples_per_task: int, selection_seed: int, temperature: float, top_p: float,
    max_tokens: int, sample_path: Path, resume: bool = False,
    model_revision: str | None = None,
) -> list[dict]:
    if max_tasks <= 0 or samples_per_task <= 0:
        raise ValueError("max_tasks and samples_per_task must be positive")
    wait_for_model_server(model_url, timeout_sec=30)
    paths = sorted(tasks_dir.glob("*.json"))
    random.Random(selection_seed).shuffle(paths)
    paths = paths[:max_tasks]
    if not paths:
        raise ValueError(f"No JSON task files in {tasks_dir}")
    selected_task_hash = hashlib.sha256()
    for path in paths:
        selected_task_hash.update(path.name.encode("utf-8"))
        selected_task_hash.update(path.read_bytes())

    config = {
        "tasks_dir": str(tasks_dir.resolve()), "model_url": model_url,
        "model_name": model_name, "max_tasks": max_tasks,
        "model_revision": model_revision,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples_per_task": samples_per_task, "selection_seed": selection_seed,
        "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        "selected_tasks_sha256": selected_task_hash.hexdigest(),
    }
    config_path = sample_path.with_name("sampling_config.json")
    if resume:
        if not config_path.is_file():
            raise ValueError(f"Cannot resume without {config_path}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume parameters do not match sampling_config.json")
        rows = _read_jsonl(sample_path) if sample_path.is_file() else []
    else:
        if sample_path.exists() or config_path.exists():
            raise ValueError("Sampling output already exists; use --resume or a new output_dir")
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        rows = []
    completed = {(row["task_id"], row["sample_index"]) for row in rows}
    if len(completed) != len(rows):
        raise ValueError("Duplicate task/sample keys in existing samples.jsonl")

    actor = LLMActor(LLMClient(api_url=chat_endpoint(model_url), model=model_name))
    with sample_path.open("a", encoding="utf-8") as output:
        for task_number, path in enumerate(paths, 1):
            task = json.loads(path.read_text(encoding="utf-8"))
            collector = ConversationCollector(
                multi_hop=task.get("retrieval_scope") == "split_corpus"
            )
            collector.add_user_message(task["user_query"])
            prompt = collector.get_prompt_for_generation()
            for sample_index in range(samples_per_task):
                if (task["task_id"], sample_index) in completed:
                    continue
                turn = actor.decide(
                    prompt, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, stop_tokens=["<|im_end|>"],
                )
                action = turn.action
                valid_schema, _ = action.validate()
                query = action.params.get("query")
                topk = action.params.get("topk", 10)
                is_search = (
                    valid_schema and action.tool == "SEARCH"
                    and isinstance(query, str) and bool(query.strip())
                    and isinstance(topk, int) and not isinstance(topk, bool)
                    and 1 <= topk <= 100
                )
                row = {
                    "task_id": task["task_id"], "sample_index": sample_index,
                    "tool": action.tool, "query": query if is_search else None,
                    "topk": topk if is_search else None,
                    "valid_search": bool(is_search), "raw_content": turn.raw_content,
                    "prompt_tokens": turn.prompt_tokens,
                    "completion_tokens": turn.completion_tokens,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                rows.append(row)
            print(f"[{task_number}/{len(paths)}] {task['task_id']}: "
                  f"{samples_per_task} actions available", flush=True)
    return rows


def analyze_samples(rows: list[dict], tasks_dir: Path, corpus_dir: Path,
                    *, jaccard_threshold: float = 0.8) -> tuple[dict, list[dict]]:
    if not 0 <= jaccard_threshold <= 1:
        raise ValueError("jaccard_threshold must be in [0, 1]")
    if not rows:
        raise ValueError("No model samples to analyze")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)

    corpus = CorpusStore(str(corpus_dir))
    corpus.load()
    corpus_count = len(corpus)
    if not corpus_count:
        raise ValueError(f"Empty corpus: {corpus_dir}")
    analyzed: list[dict] = []
    per_task: list[dict] = []
    try:
        for task_id, task_rows in sorted(by_task.items()):
            task_path = tasks_dir / f"{task_id}.json"
            if not task_path.is_file():
                raise ValueError(f"Missing task for sample group: {task_path}")
            task = json.loads(task_path.read_text(encoding="utf-8"))
            proposals: list[SearchProposal] = []
            gold = set(task.get("ground_truth_citations", []))
            first_hop = ((task.get("analysis") or {}).get("hops") or [{}])[0].get(
                "supporting_chunk_id"
            )
            valid_results: list[dict] = []
            for row in task_rows:
                record = {"task_id": task_id, "sample_index": row["sample_index"],
                          "query": row.get("query"), "valid_search": False}
                if row.get("valid_search") and isinstance(row.get("query"), str):
                    query = row["query"]
                    topk = row.get("topk", 10)
                    if (not isinstance(topk, int) or isinstance(topk, bool)
                            or not 1 <= topk <= 100):
                        raise ValueError(f"Invalid topk in {task_id} sample {row['sample_index']}")
                    candidates = corpus.search(
                        query, topk=topk, doc_ids=task.get("reference_docs") or None
                    )
                    effect = effect_key(item.chunk_id for item in candidates)
                    proposals.append(SearchProposal(query=query, effect=effect))
                    record.update({
                        "valid_search": True, "topk": topk,
                        "effect_chunk_ids": list(effect),
                        "candidate_scores": [float(item.score) for item in candidates],
                        "gold_recall": len(gold & set(effect)) / len(gold) if gold else None,
                        "first_hop_hit": first_hop in effect if first_hop else None,
                    })
                    valid_results.append(record)
                analyzed.append(record)

            groups = group_by_effect(proposals)
            surfaces = {(" ".join(p.query.casefold().split()), record["topk"])
                        for p, record in zip(proposals, valid_results)}
            # Distinct query texts can produce the same ordered observation.
            effects = list(groups)
            if len(groups) > len(surfaces):
                raise ValueError(f"Non-deterministic retrieval for repeated SEARCH actions in {task_id}")
            # The SEARCH observation hides retrieval scores, but EnvState keeps
            # them and RERANK can use them. Same ordered IDs are therefore not
            # automatically the same post-tool state.
            score_spreads = []
            for indices in groups.values():
                distinct = {(" ".join(proposals[index].query.casefold().split()),
                             valid_results[index]["topk"]): index for index in indices}
                representatives = list(distinct.values())
                if len(representatives) < 2:
                    continue
                vectors = [valid_results[index]["candidate_scores"]
                           for index in representatives]
                score_spreads.append(max(
                    abs(left_score - right_score)
                    for left_index, left in enumerate(vectors)
                    for right in vectors[left_index + 1:]
                    for left_score, right_score in zip(left, right)
                ) if vectors[0] else 0.0)
            pairs = [(effects[i], effects[j]) for i in range(len(effects))
                     for j in range(i + 1, len(effects))]
            per_task.append({
                "task_id": task_id, "hop_count": (task.get("analysis") or {}).get("hop_count"),
                "samples": len(task_rows), "valid_searches": len(proposals),
                "unique_queries": len(surfaces), "unique_ordered_effects": len(groups),
                "unique_unordered_effects": len({frozenset(effect) for effect in groups}),
                "duplicate_surface_samples": len(proposals) - len(surfaces),
                "duplicate_effect_samples": len(proposals) - len(groups),
                "distinct_query_effect_collisions": len(surfaces) - len(groups),
                "distinct_query_same_ids_classes": len(score_spreads),
                "same_ids_classes_with_score_shift": sum(
                    spread > 1e-9 for spread in score_spreads
                ),
                "max_same_ids_score_shift": max(score_spreads, default=0.0),
                "near_effect_pairs": sum(jaccard(a, b) >= jaccard_threshold for a, b in pairs),
                "distinct_effect_pairs": len(pairs),
                "mean_gold_recall": (
                    sum(record["gold_recall"] for record in valid_results) / len(valid_results)
                    if valid_results and gold else None
                ),
                "any_first_hop_hit": any(record["first_hop_hit"] for record in valid_results),
            })
    finally:
        corpus.close()

    totals = {key: sum(item[key] for item in per_task) for key in (
        "samples", "valid_searches", "unique_queries", "unique_ordered_effects",
        "duplicate_surface_samples", "duplicate_effect_samples",
        "distinct_query_effect_collisions", "distinct_query_same_ids_classes",
        "same_ids_classes_with_score_shift", "near_effect_pairs", "distinct_effect_pairs",
    )}
    valid = totals["valid_searches"]
    by_hop: dict[str, dict] = {}
    for hop_count in sorted({str(item["hop_count"]) for item in per_task}):
        subset = [item for item in per_task if str(item["hop_count"]) == hop_count]
        distinct_queries = sum(item["unique_queries"] for item in subset)
        by_hop[hop_count] = {
            "tasks": len(subset),
            "valid_searches": sum(item["valid_searches"] for item in subset),
            "unique_queries": distinct_queries,
            "unique_ordered_effects": sum(item["unique_ordered_effects"] for item in subset),
            "distinct_query_effect_collision_rate": (
                sum(item["distinct_query_effect_collisions"] for item in subset)
                / distinct_queries if distinct_queries else None
            ),
        }
    summary = {
        "protocol": "first-search shared-prefix exact-effect pilot v1",
        "tasks": len(per_task), "corpus_chunks": corpus_count,
        "surface_definition": "casefolded whitespace-normalized query plus topk",
        "jaccard_threshold_diagnostic_only": jaccard_threshold,
        **totals,
        "valid_search_rate": valid / totals["samples"],
        "exact_effect_duplicate_rate": totals["duplicate_effect_samples"] / valid if valid else None,
        "distinct_query_effect_collision_rate": (
            totals["distinct_query_effect_collisions"] / totals["unique_queries"]
            if totals["unique_queries"] else None
        ),
        "same_ids_score_shift_class_rate": (
            totals["same_ids_classes_with_score_shift"]
            / totals["distinct_query_same_ids_classes"]
            if totals["distinct_query_same_ids_classes"] else None
        ),
        "max_same_ids_score_shift": max(
            (item["max_same_ids_score_shift"] for item in per_task), default=0.0
        ),
        "hypothetical_continuations_one_per_exact_effect": totals["unique_ordered_effects"],
        "by_hop_count": by_hop,
        "per_task": per_task,
        "limitations": [
            "This measures first SEARCH only, not later bridge-query states.",
            "Identical retrieved chunks do not imply identical post-tool states while raw query text remains in the transcript.",
            "The continuation count is hypothetical; no downstream policy-gradient or wall-clock saving is measured.",
            "Gold citations are used only for offline diagnostics, never in model prompts.",
        ],
    }
    return summary, analyzed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks_dir", type=Path, required=True)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--samples_file", type=Path,
                        help="Existing first-search model samples; skip API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted model sampling run with the same parameters")
    parser.add_argument("--model_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", default="Qwen3.5-9B")
    parser.add_argument("--model_revision",
                        help="Checkpoint path/revision actually loaded by the model service")
    parser.add_argument("--max_tasks", type=int, default=200)
    parser.add_argument("--samples_per_task", type=int, default=8)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--near_jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.temperature <= 0 and not args.samples_file:
        parser.error("stochastic sampling requires --temperature > 0")
    if args.resume and args.samples_file:
        parser.error("--resume applies to model sampling, not --samples_file")
    if not args.tasks_dir.is_dir() or not args.corpus_dir.is_dir():
        parser.error("tasks_dir and corpus_dir must be existing directories")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.samples_file:
        rows = _read_jsonl(args.samples_file)
    else:
        print("Pilot purpose: measure exact retrieval-effect collisions under a shared prefix.", flush=True)
        rows = sample_first_search(
            args.tasks_dir, args.model_url, args.model_name,
            max_tasks=args.max_tasks, samples_per_task=args.samples_per_task,
            selection_seed=args.selection_seed, temperature=args.temperature,
            top_p=args.top_p, max_tokens=args.max_tokens,
            sample_path=args.output_dir / "samples.jsonl", resume=args.resume,
            model_revision=args.model_revision,
        )
    if args.samples_file:
        _write_jsonl(args.output_dir / "samples.jsonl", rows)
    summary, analyzed = analyze_samples(
        rows, args.tasks_dir, args.corpus_dir, jaccard_threshold=args.near_jaccard
    )
    summary["sampling"] = {
        "source": str(args.samples_file) if args.samples_file else "model_api",
        "model_name": args.model_name if not args.samples_file else None,
        "model_revision": args.model_revision if not args.samples_file else None,
        "temperature": args.temperature if not args.samples_file else None,
        "top_p": args.top_p if not args.samples_file else None,
        "selection_seed": args.selection_seed if not args.samples_file else None,
    }
    _write_jsonl(args.output_dir / "effect_records.jsonl", analyzed)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {key: value for key, value in summary.items() if key != "per_task"}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
