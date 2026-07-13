"""Create reproducible, task-isolated HotpotQA train/eval benchmarks.

The HotpotQA ``distractor`` configuration provides each question with a small
set of Wikipedia paragraphs containing both evidence and distractors.  We keep
those candidate paragraphs local to the task: this prevents a train item (or a
different evaluation item) from becoming an accidental retrieval source.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


TRAIN_FRACTION = 0.7


def sanitize_title(title: str) -> str:
    """Return a stable filename-safe title fragment."""
    return "".join(char if char.isalnum() else "_" for char in title).lower()


def partition_samples(
    samples: List[Dict[str, Any]], train_count: int, eval_count: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select and partition samples without task-id overlap."""
    required = train_count + eval_count
    if len(samples) < required:
        raise ValueError(f"Need {required} samples, received {len(samples)}")

    selected = samples[:required]
    train_samples = selected[:train_count]
    eval_samples = selected[train_count:]
    train_ids = {item["id"] for item in train_samples}
    eval_ids = {item["id"] for item in eval_samples}
    if not train_ids.isdisjoint(eval_ids):
        raise ValueError("Train and eval task IDs overlap")
    return train_samples, eval_samples


def _reset_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Pass --overwrite to replace this generated benchmark."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _write_tasks_and_corpus(
    samples: Iterable[Dict[str, Any]], tasks_dir: Path, corpus_dir: Path
) -> tuple[List[Dict[str, Any]], int]:
    """Write one split and scope every task's search space to its own context."""
    tasks: List[Dict[str, Any]] = []
    chunk_count = 0

    for item in samples:
        task_id = f"hotpot_{item['id']}"
        context = item["context"]
        supporting_titles = {fact[0] for fact in item["supporting_facts"]}
        citations: List[str] = []

        for title, sentences in zip(context["title"], context["sentences"]):
            chunk_id = f"{sanitize_title(title)}_{task_id}"
            chunk_path = corpus_dir / f"{chunk_id}.json"
            text_content = " ".join(sentences)
            with chunk_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": task_id,
                        "title": title,
                        "content": text_content,
                        "metadata": {"task_source": "hotpot_qa", "split_task_id": task_id},
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            chunk_count += 1
            if title in supporting_titles:
                citations.append(chunk_id)

        if not citations:
            raise ValueError(f"{task_id} has no supporting-title citations")

        task = {
            "task_id": task_id,
            "user_query": item["question"],
            "ground_truth_answer": item["answer"],
            "ground_truth_citations": citations,
            # CorpusStore filters on doc_id, so a task cannot retrieve another
            # task's context even when the whole split corpus is loaded.
            "reference_docs": [task_id],
        }
        with (tasks_dir / f"{task_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(task, handle, ensure_ascii=False, indent=2)
        tasks.append(task)

    return tasks, chunk_count


def build_split(
    samples: List[Dict[str, Any]],
    output_dir: str | Path,
    train_count: int,
    eval_count: int,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build a train/eval split with isolated corpora and a manifest."""
    output_path = Path(output_dir)
    _reset_output_dir(output_path, overwrite=overwrite)
    train_samples, eval_samples = partition_samples(samples, train_count, eval_count)

    train_tasks_dir = output_path / "tasks" / "train"
    eval_tasks_dir = output_path / "tasks" / "eval"
    train_corpus_dir = output_path / "corpus" / "train"
    eval_corpus_dir = output_path / "corpus" / "eval"
    for directory in (train_tasks_dir, eval_tasks_dir, train_corpus_dir, eval_corpus_dir):
        directory.mkdir(parents=True, exist_ok=False)

    print(f"Processing {len(train_samples)} training tasks in {output_path}...")
    train_tasks, train_chunks = _write_tasks_and_corpus(
        train_samples, train_tasks_dir, train_corpus_dir
    )
    print(f"Processing {len(eval_samples)} evaluation tasks in {output_path}...")
    eval_tasks, eval_chunks = _write_tasks_and_corpus(eval_samples, eval_tasks_dir, eval_corpus_dir)

    train_ids = {task["task_id"] for task in train_tasks}
    eval_ids = {task["task_id"] for task in eval_tasks}
    if not train_ids.isdisjoint(eval_ids):
        raise AssertionError("Train and eval task IDs overlap")

    all_citations_exist = all(
        (train_corpus_dir / f"{citation}.json").exists()
        for task in train_tasks
        for citation in task["ground_truth_citations"]
    ) and all(
        (eval_corpus_dir / f"{citation}.json").exists()
        for task in eval_tasks
        for citation in task["ground_truth_citations"]
    )
    if not all_citations_exist:
        raise AssertionError("A task references a missing citation chunk")

    total_tasks = len(train_tasks) + len(eval_tasks)
    stats = {
        "split_name": output_path.name,
        "total_tasks": total_tasks,
        "train_tasks_count": len(train_tasks),
        "eval_tasks_count": len(eval_tasks),
        "train_fraction": len(train_tasks) / total_tasks if total_tasks else 0.0,
        "eval_fraction": len(eval_tasks) / total_tasks if total_tasks else 0.0,
        "train_corpus_chunks": train_chunks,
        "eval_corpus_chunks": eval_chunks,
        "overlap_verification": {
            "task_id_overlap_detected": False,
            "corpus_isolated_by_split": True,
            "task_retrieval_scoped_by_reference_docs": True,
            "all_citations_exist": True,
        },
    }
    with (output_path / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return stats


def _load_hotpotqa_validation() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The optional benchmark dependency is missing. Install it with "
            "`python -m pip install 'datasets>=2.18'`."
        ) from exc

    return load_dataset("hotpot_qa", "distractor", split="validation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create task-isolated 7:3 train/eval HotpotQA distractor benchmarks."
    )
    parser.add_argument("--output_root", default="data", help="Directory in which benchmark folders are created")
    parser.add_argument("--seed", type=int, default=42, help="Dataset shuffle seed")
    parser.add_argument("--debug_total", type=int, default=20, help="Total tasks in hotpotqa_debug")
    parser.add_argument("--mini_total", type=int, default=130, help="Total tasks in hotpotqa_mini")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated output folders")
    args = parser.parse_args()

    if args.debug_total <= 0 or args.mini_total <= 0:
        parser.error("--debug_total and --mini_total must be positive")

    print("=" * 60)
    print("ResearchAgent-RL: HotpotQA 7:3 Benchmark Setup")
    print("=" * 60)
    raw_dataset = _load_hotpotqa_validation()

    total_needed = args.debug_total + args.mini_total
    if len(raw_dataset) < total_needed:
        raise RuntimeError(f"Dataset has {len(raw_dataset)} items; need {total_needed}")
    indices = list(range(len(raw_dataset)))
    random.Random(args.seed).shuffle(indices)
    samples = [raw_dataset[index] for index in indices[:total_needed]]

    debug_train = round(args.debug_total * TRAIN_FRACTION)
    mini_train = round(args.mini_total * TRAIN_FRACTION)
    debug_stats = build_split(
        samples[: args.debug_total],
        Path(args.output_root) / "hotpotqa_debug",
        train_count=debug_train,
        eval_count=args.debug_total - debug_train,
        overwrite=args.overwrite,
    )
    mini_stats = build_split(
        samples[args.debug_total :],
        Path(args.output_root) / "hotpotqa_mini",
        train_count=mini_train,
        eval_count=args.mini_total - mini_train,
        overwrite=args.overwrite,
    )

    print("\n[SUCCESS] HotpotQA benchmark generation completed.")
    print(
        f"- Debug: {debug_stats['train_tasks_count']} train / "
        f"{debug_stats['eval_tasks_count']} eval"
    )
    print(
        f"- Mini: {mini_stats['train_tasks_count']} train / "
        f"{mini_stats['eval_tasks_count']} eval"
    )


if __name__ == "__main__":
    main()
