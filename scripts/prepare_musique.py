#!/usr/bin/env python3
"""Build a pooled, indexed MuSiQue-Answerable benchmark for Agent RL.

Input files are the official ``musique_ans_v1.0_{train,dev}.jsonl`` releases.
The labeled official dev split is exposed as ``eval``; the official test file
cannot be scored locally because its answers are hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Iterator


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def _paragraph_id(paragraph: dict[str, Any]) -> str:
    title = paragraph["title"].strip()
    text = paragraph["paragraph_text"].strip()
    if not title or not text:
        raise ValueError("MuSiQue paragraph has an empty title or text")
    digest = hashlib.sha256(f"{title}\0{text}".encode("utf-8")).hexdigest()[:24]
    return f"musique_{digest}"


def _open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE chunks ("
        "rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, "
        "doc_id TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE VIRTUAL TABLE chunk_fts USING fts5("
        "title, content, content='chunks', content_rowid='rowid', "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return connection


def _insert_paragraph(connection: sqlite3.Connection, paragraph: dict[str, Any]) -> str:
    chunk_id = _paragraph_id(paragraph)
    title = paragraph["title"].strip()
    content = paragraph["paragraph_text"].strip()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO chunks (chunk_id, doc_id, title, content) VALUES (?, ?, ?, ?)",
        (chunk_id, chunk_id, title, content),
    )
    if cursor.rowcount:
        connection.execute(
            "INSERT INTO chunk_fts (rowid, title, content) VALUES (?, ?, ?)",
            (cursor.lastrowid, title, content),
        )
    return chunk_id


def _validate_record(row: dict[str, Any], split: str) -> tuple[str, list[dict[str, Any]]]:
    raw_id = row.get("id")
    if not isinstance(raw_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", raw_id):
        raise ValueError(f"{split}: unsafe or missing MuSiQue id: {raw_id!r}")
    if row.get("answerable") is not True or not row.get("answer") or not row.get("question"):
        raise ValueError(f"{split}:{raw_id}: requires a labeled answerable record")
    decomposition = row.get("question_decomposition")
    if not isinstance(decomposition, list) or not 2 <= len(decomposition) <= 4:
        raise ValueError(f"{split}:{raw_id}: expected a 2–4 hop decomposition")
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError(f"{split}:{raw_id}: missing paragraphs")
    indices = [paragraph.get("idx") for paragraph in paragraphs]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{split}:{raw_id}: duplicate paragraph idx")
    supported = {paragraph["idx"] for paragraph in paragraphs if paragraph.get("is_supporting") is True}
    if len(supported) < 2:
        raise ValueError(f"{split}:{raw_id}: fewer than two supporting paragraphs")
    for hop in decomposition:
        if hop.get("paragraph_support_idx") not in supported:
            raise ValueError(f"{split}:{raw_id}: a hop has no supporting paragraph")
    return raw_id, paragraphs


def _build_split(source: Path, split: str, output: Path,
                 excluded_ids: set[str] | None = None) -> dict[str, Any]:
    tasks_dir = output / "tasks" / split
    corpus_dir = output / "corpus" / split
    tasks_dir.mkdir(parents=True)
    corpus_dir.mkdir(parents=True)
    connection = _open_index(corpus_dir / "corpus.sqlite")
    task_ids: set[str] = set()
    hop_counts: Counter[int] = Counter()
    supporting_ids: set[str] = set()
    excluded_seen: set[str] = set()
    try:
        for row in _records(source):
            raw_id, paragraphs = _validate_record(row, split)
            task_id = f"musique_{raw_id}"
            if task_id in task_ids:
                raise ValueError(f"{split}: duplicate task id {task_id}")
            task_ids.add(task_id)
            by_idx = {}
            for paragraph in paragraphs:
                chunk_id = _paragraph_id(paragraph)
                if excluded_ids and chunk_id in excluded_ids:
                    if paragraph["is_supporting"]:
                        raise ValueError(
                            f"{split}:{raw_id}: supporting paragraph overlaps evaluation gold"
                        )
                    excluded_seen.add(chunk_id)
                    continue
                by_idx[paragraph["idx"]] = _insert_paragraph(connection, paragraph)
            citations = list(dict.fromkeys(
                by_idx[paragraph["idx"]]
                for paragraph in paragraphs if paragraph["is_supporting"]
            ))
            supporting_ids.update(citations)
            hops = [
                {"question": hop["question"], "answer": hop["answer"],
                 "supporting_chunk_id": by_idx[hop["paragraph_support_idx"]]}
                for hop in row["question_decomposition"]
            ]
            hop_counts[len(hops)] += 1
            task = {
                "task_id": task_id,
                "user_query": row["question"],
                "ground_truth_answer": row["answer"],
                "ground_truth_answer_aliases": row.get("answer_aliases") or [],
                "ground_truth_citations": citations,
                "reference_docs": [],
                "retrieval_scope": "split_corpus",
                "analysis": {"hop_count": len(hops), "hops": hops,
                             "source_split": "dev" if split == "eval" else "train"},
            }
            (tasks_dir / f"{task_id}.json").write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if len(task_ids) % 1000 == 0:
                connection.commit()
                print(f"{split}: {len(task_ids)} tasks indexed", flush=True)
        connection.execute(
            "INSERT INTO index_metadata (key, value) VALUES ('schema_version', '1')"
        )
        connection.execute(
            "INSERT INTO index_metadata (key, value) VALUES ('complete', '1')"
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        connection.close()
    return {
        "tasks": len(task_ids), "chunks": chunk_count,
        "hop_counts": dict(sorted(hop_counts.items())),
        "task_ids": task_ids, "supporting_ids": supporting_ids,
        "excluded_unique_passages": len(excluded_seen),
    }


def _evaluation_support_ids(dev_source: Path) -> set[str]:
    ids: set[str] = set()
    for row in _records(dev_source):
        _validate_record(row, "eval")
        ids.update(
            _paragraph_id(paragraph)
            for paragraph in row["paragraphs"] if paragraph["is_supporting"]
        )
    return ids


def build_benchmark(train_file: str | Path, dev_file: str | Path, output_dir: str | Path) -> dict[str, Any]:
    train_source, dev_source, output = Path(train_file), Path(dev_file), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; use a new directory")
    if not train_source.is_file() or not dev_source.is_file():
        raise FileNotFoundError("Both official MuSiQue train and dev JSONL files are required")
    if train_source.resolve() == dev_source.resolve():
        raise ValueError("Train and dev must be different source files")
    eval_gold_ids = _evaluation_support_ids(dev_source)
    output.mkdir(parents=True)
    train = _build_split(train_source, "train", output, excluded_ids=eval_gold_ids)
    evaluation = _build_split(dev_source, "eval", output)
    if train["task_ids"] & evaluation["task_ids"]:
        raise ValueError("Official MuSiQue train/dev task IDs overlap")
    with closing(sqlite3.connect(output / "corpus" / "train" / "corpus.sqlite")) as connection:
        eval_gold_in_train = sum(
            connection.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            is not None
            for chunk_id in eval_gold_ids
        )
    if eval_gold_in_train:
        raise AssertionError("Evaluation gold passages leaked into the training corpus")
    stats = {
        "format": "research-agent-musique-rl-v1",
        "source": {"dataset": "MuSiQue-Answerable v1.0", "train_split": "train",
                   "eval_split": "dev", "official_test_labels_available": False,
                   "train_sha256": _sha256_file(train_source),
                   "dev_sha256": _sha256_file(dev_source)},
        "retrieval_scope": "split_corpus",
        "train_tasks_count": train["tasks"], "eval_tasks_count": evaluation["tasks"],
        "train_corpus_chunks": train["chunks"],
        "eval_corpus_chunks": evaluation["chunks"],
        "train_hop_counts": train["hop_counts"], "eval_hop_counts": evaluation["hop_counts"],
        "task_id_overlap_count": 0,
        "supporting_passage_overlap_count": len(train["supporting_ids"] & evaluation["supporting_ids"]),
        "eval_gold_excluded_from_train_corpus": train["excluded_unique_passages"],
        "eval_gold_present_in_train_corpus": eval_gold_in_train,
    }
    (output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare indexed MuSiQue-Ans train/eval for Agent RL")
    parser.add_argument("--source_dir", type=Path, required=True,
                        help="Directory containing official MuSiQue-Ans v1.0 JSONL files")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    train = args.source_dir / "musique_ans_v1.0_train.jsonl"
    dev = args.source_dir / "musique_ans_v1.0_dev.jsonl"
    stats = build_benchmark(train, dev, args.output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
