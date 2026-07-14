#!/usr/bin/env python3
"""Convert ResearchAgent task JSON files into Slime prompt JSONL.

The output keeps the verifier fields in ``metadata`` so a custom Slime
generation/reward plugin can reconstruct a task-local ResearchEnv rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def _task_paths(tasks_dir: Path) -> list[Path]:
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Tasks directory does not exist: {tasks_dir}")
    paths = sorted(path for path in tasks_dir.iterdir() if path.suffix == ".json")
    if not paths:
        raise ValueError(f"No task JSON files found in: {tasks_dir}")
    return paths


def build_records(
    tasks_dir: str | Path,
    *,
    max_samples: int | None = None,
    selection_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, validate, and deterministically select Slime prompt records."""
    paths = _task_paths(Path(tasks_dir))
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when specified")
    if selection_seed is not None:
        random.Random(selection_seed).shuffle(paths)
    if max_samples is not None:
        paths = paths[:max_samples]

    records: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    digest = hashlib.sha256()
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        task_id = raw["task_id"]
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id in selected input: {task_id}")
        task_ids.add(task_id)
        if not raw.get("user_query"):
            raise ValueError(f"{task_id} has an empty user_query")
        if not raw.get("ground_truth_answer"):
            raise ValueError(f"{task_id} has an empty ground_truth_answer")
        if not raw.get("ground_truth_citations"):
            raise ValueError(f"{task_id} has no ground_truth_citations")
        if not raw.get("reference_docs"):
            raise ValueError(f"{task_id} has no reference_docs for retrieval scoping")

        metadata = {
            "task_id": task_id,
            "user_query": raw["user_query"],
            "ground_truth_answer": raw["ground_truth_answer"],
            "ground_truth_citations": raw["ground_truth_citations"],
            "reference_docs": raw["reference_docs"],
        }
        record = {
            # A string prompt avoids ambiguity at the Slime input boundary;
            # custom_generate builds the tool-use system prompt itself.
            "prompt": raw["user_query"],
            "label": raw["ground_truth_answer"],
            "metadata": metadata,
        }
        records.append(record)
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")

    manifest = {
        "format": "research-agent-slime-v1",
        "tasks_dir": str(Path(tasks_dir).resolve()),
        "records": len(records),
        "selection_seed": selection_seed,
        "max_samples": max_samples,
        "task_ids_sha256": hashlib.sha256("\n".join(sorted(task_ids)).encode("utf-8")).hexdigest(),
        "records_sha256": digest.hexdigest(),
    }
    return records, manifest


def write_dataset(
    records: list[dict[str, Any]], manifest: dict[str, Any], output_file: str | Path
) -> tuple[Path, Path]:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ResearchAgent task JSON files into Slime JSONL.")
    parser.add_argument("--tasks_dir", required=True, help="Directory containing train task JSON files")
    parser.add_argument("--output_file", required=True, help="Output prompt JSONL path")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional deterministic subset size")
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=None,
        help="Shuffle task files with this seed before applying --max_samples",
    )
    args = parser.parse_args()
    try:
        records, manifest = build_records(
            args.tasks_dir, max_samples=args.max_samples, selection_seed=args.selection_seed
        )
        output_path, manifest_path = write_dataset(records, manifest, args.output_file)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(f"Generated {len(records)} Slime prompt records: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
