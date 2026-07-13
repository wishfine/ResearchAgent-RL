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
from typing import Any, Dict, Iterable, List, Set


DEFAULT_TRAIN_TOTAL = 7_000
DEFAULT_EVAL_TOTAL = 3_000


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


def select_samples(dataset: Any, count: int, seed: int) -> List[Dict[str, Any]]:
    """Select a deterministic subset from one official HotpotQA split."""
    if count <= 0:
        raise ValueError("Sample count must be positive")
    if len(dataset) < count:
        raise ValueError(f"Dataset has {len(dataset)} items; need {count}")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return [dataset[index] for index in indices[:count]]


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
) -> tuple[List[Dict[str, Any]], int, Set[str]]:
    """Write one split with one corpus document file per task.

    A distractor task has roughly ten candidate paragraphs.  Keeping them in a
    single ``Document`` JSON preserves the task-level ``doc_id`` filter while
    avoiding ten tiny files per task (and the resulting inode pressure).
    """
    tasks: List[Dict[str, Any]] = []
    chunk_count = 0
    chunk_ids: Set[str] = set()

    for item in samples:
        task_id = f"hotpot_{item['id']}"
        context = item["context"]
        supporting_titles = {fact[0] for fact in item["supporting_facts"]}
        citations: List[str] = []
        chunks: List[Dict[str, str]] = []

        for title, sentences in zip(context["title"], context["sentences"]):
            chunk_id = f"{sanitize_title(title)}_{task_id}"
            text_content = " ".join(sentences)
            chunks.append({"chunk_id": chunk_id, "title": title, "content": text_content})
            chunk_ids.add(chunk_id)
            chunk_count += 1
            if title in supporting_titles:
                citations.append(chunk_id)

        if not citations:
            raise ValueError(f"{task_id} has no supporting-title citations")

        with (corpus_dir / f"{task_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "doc_id": task_id,
                    "title": task_id,
                    "metadata": {"task_source": "hotpot_qa", "split_task_id": task_id},
                    "chunks": chunks,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

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

    return tasks, chunk_count, chunk_ids


def build_benchmark(
    train_samples: List[Dict[str, Any]],
    eval_samples: List[Dict[str, Any]],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    source_splits: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Build a train/eval benchmark with isolated corpora and a manifest."""
    raw_train_ids = {item["id"] for item in train_samples}
    raw_eval_ids = {item["id"] for item in eval_samples}
    if not raw_train_ids.isdisjoint(raw_eval_ids):
        raise ValueError("Train and eval task IDs overlap")

    output_path = Path(output_dir)
    _reset_output_dir(output_path, overwrite=overwrite)

    train_tasks_dir = output_path / "tasks" / "train"
    eval_tasks_dir = output_path / "tasks" / "eval"
    train_corpus_dir = output_path / "corpus" / "train"
    eval_corpus_dir = output_path / "corpus" / "eval"
    for directory in (train_tasks_dir, eval_tasks_dir, train_corpus_dir, eval_corpus_dir):
        directory.mkdir(parents=True, exist_ok=False)

    print(f"Processing {len(train_samples)} training tasks in {output_path}...")
    train_tasks, train_chunks, train_chunk_ids = _write_tasks_and_corpus(
        train_samples, train_tasks_dir, train_corpus_dir
    )
    print(f"Processing {len(eval_samples)} evaluation tasks in {output_path}...")
    eval_tasks, eval_chunks, eval_chunk_ids = _write_tasks_and_corpus(
        eval_samples, eval_tasks_dir, eval_corpus_dir
    )

    train_ids = {task["task_id"] for task in train_tasks}
    eval_ids = {task["task_id"] for task in eval_tasks}
    if not train_ids.isdisjoint(eval_ids):
        raise AssertionError("Train and eval task IDs overlap")

    all_citations_exist = all(
        citation in train_chunk_ids
        for task in train_tasks
        for citation in task["ground_truth_citations"]
    ) and all(
        citation in eval_chunk_ids
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
        "train_corpus_document_files": len(train_tasks),
        "eval_corpus_document_files": len(eval_tasks),
        "source": {
            "dataset": "hotpot_qa",
            "configuration": "distractor",
            "splits": source_splits or {"train": "unspecified", "eval": "unspecified"},
        },
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


def build_split(
    samples: List[Dict[str, Any]],
    output_dir: str | Path,
    train_count: int,
    eval_count: int,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build a random 7:3-style split; retained for small local smoke tests."""
    train_samples, eval_samples = partition_samples(samples, train_count, eval_count)
    return build_benchmark(
        train_samples,
        eval_samples,
        output_dir,
        overwrite=overwrite,
        source_splits={"train": "custom", "eval": "custom"},
    )


def _load_hotpotqa_split(split: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The optional benchmark dependency is missing. Install it with "
            "`python -m pip install 'datasets>=2.18'`."
        ) from exc

    return load_dataset("hotpot_qa", "distractor", split=split)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a task-isolated HotpotQA benchmark from official train and validation splits."
    )
    parser.add_argument(
        "--output_dir",
        default="data/hotpotqa_7k3k",
        help="Directory for the generated benchmark",
    )
    parser.add_argument("--seed", type=int, default=42, help="Dataset shuffle seed")
    parser.add_argument(
        "--hf_endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Hugging Face endpoint used while downloading the dataset",
    )
    parser.add_argument(
        "--train_total",
        type=int,
        default=DEFAULT_TRAIN_TOTAL,
        help="Tasks sampled from the official HotpotQA train split",
    )
    parser.add_argument(
        "--eval_total",
        type=int,
        default=DEFAULT_EVAL_TOTAL,
        help="Tasks sampled from the official HotpotQA validation split",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated output folders")
    args = parser.parse_args()

    if args.train_total <= 0 or args.eval_total <= 0:
        parser.error("--train_total and --eval_total must be positive")

    print("=" * 60)
    print("ResearchAgent-RL: HotpotQA 7:3 Benchmark Setup")
    print("=" * 60)
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    print(f"Loading hotpot_qa/distractor official train and validation via {args.hf_endpoint} ...")
    official_train = _load_hotpotqa_split("train")
    official_eval = _load_hotpotqa_split("validation")
    train_samples = select_samples(official_train, args.train_total, seed=args.seed)
    eval_samples = select_samples(official_eval, args.eval_total, seed=args.seed + 1)

    stats = build_benchmark(
        train_samples,
        eval_samples,
        args.output_dir,
        overwrite=args.overwrite,
        source_splits={"train": "train", "eval": "validation"},
    )

    print("\n[SUCCESS] HotpotQA benchmark generation completed.")
    print(
        f"- {stats['train_tasks_count']} train / {stats['eval_tasks_count']} eval "
        f"({stats['train_fraction']:.1%} / {stats['eval_fraction']:.1%})"
    )


if __name__ == "__main__":
    main()
