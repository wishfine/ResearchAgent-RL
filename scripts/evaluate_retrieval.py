#!/usr/bin/env python3
"""Measure first-SEARCH evidence recall before spending GPU time on Agent RL."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.core.corpus.store import CorpusStore


def evaluate(tasks_dir: Path, corpus_dir: Path, topks: list[int],
             max_tasks: int | None = None, seed: int = 42) -> dict:
    if not topks or any(value <= 0 for value in topks):
        raise ValueError("topks must contain positive integers")
    paths = sorted(tasks_dir.glob("*.json"))
    if max_tasks is not None:
        random.Random(seed).shuffle(paths)
        paths = paths[:max_tasks]
    if not paths:
        raise ValueError(f"No tasks found in {tasks_dir}")
    corpus = CorpusStore(str(corpus_dir))
    corpus.load()
    if not len(corpus):
        raise ValueError(f"Empty corpus: {corpus_dir}")

    totals = {topk: {"recall_sum": 0.0, "all_support_hits": 0, "first_hop_hits": 0,
                     "oracle_hop_hits": 0, "oracle_hop_total": 0,
                     "oracle_complete_chains": 0}
              for topk in sorted(set(topks))}
    hop_counts: dict[int, int] = {}
    for path in paths:
        task = json.loads(path.read_text(encoding="utf-8"))
        gold = set(task["ground_truth_citations"])
        if not gold:
            raise ValueError(f"Task {path} has no gold citations")
        analysis = task.get("analysis", {})
        hops = analysis.get("hops", [])
        hop_count = analysis.get("hop_count", len(gold))
        hop_counts[hop_count] = hop_counts.get(hop_count, 0) + 1
        candidates = corpus.search(task["user_query"], topk=max(totals),
                                   doc_ids=task.get("reference_docs") or None)
        oracle_hop_results = []
        for hop in hops:
            # Oracle bridge queries are a retriever diagnostic only. They use
            # gold intermediate answers and must never enter model prompts.
            query = re.sub(
                r"#(\d+)",
                lambda match: hops[int(match.group(1)) - 1]["answer"]
                if 0 < int(match.group(1)) <= len(hops) else match.group(0),
                hop["question"],
            )
            oracle_hop_results.append(
                (hop["supporting_chunk_id"], corpus.search(query, topk=max(totals)))
            )
        for topk, count in totals.items():
            returned = {candidate.chunk_id for candidate in candidates[:topk]}
            count["recall_sum"] += len(gold & returned) / len(gold)
            count["all_support_hits"] += int(gold <= returned)
            if hops:
                count["first_hop_hits"] += int(hops[0]["supporting_chunk_id"] in returned)
                hop_hits = [gold_id in {candidate.chunk_id for candidate in results[:topk]}
                            for gold_id, results in oracle_hop_results]
                count["oracle_hop_hits"] += sum(hop_hits)
                count["oracle_hop_total"] += len(hop_hits)
                count["oracle_complete_chains"] += int(all(hop_hits))
    result = {
        "tasks": len(paths), "corpus_chunks": len(corpus),
        "hop_counts": dict(sorted(hop_counts.items())),
        "single_search": {
            str(topk): {
                "support_recall": count["recall_sum"] / len(paths),
                "all_support_recall": count["all_support_hits"] / len(paths),
                "first_hop_recall": count["first_hop_hits"] / len(paths),
                "oracle_hop_recall": (
                    count["oracle_hop_hits"] / count["oracle_hop_total"]
                    if count["oracle_hop_total"] else None
                ),
                "oracle_complete_chain_recall": count["oracle_complete_chains"] / len(paths),
            }
            for topk, count in totals.items()
        },
    }
    corpus.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexical evidence retrieval")
    parser.add_argument("--tasks_dir", type=Path, required=True)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max_tasks", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=Path)
    args = parser.parse_args()
    result = evaluate(args.tasks_dir, args.corpus_dir, args.topk,
                      max_tasks=args.max_tasks, seed=args.seed)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
