from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.core.corpus.store import CorpusStore
from scripts.prepare_hotpotqa import (
    HOTPOTQA_REPO_ID,
    _load_hotpotqa_split,
    build_benchmark,
    build_split,
    partition_samples,
    select_samples,
    supporting_fact_titles,
)


def sample(index: int) -> dict:
    return {
        "id": f"sample_{index}",
        "question": f"Question {index}?",
        "answer": f"Answer {index}",
        "context": {
            "title": [f"Evidence {index}", f"Distractor {index}"],
            "sentences": [[f"Evidence sentence {index}."], [f"Distractor sentence {index}."]],
        },
        "supporting_facts": [[f"Evidence {index}", 0]],
    }


class TestHotpotQAPreparation(unittest.TestCase):
    def test_supporting_fact_titles_supports_raw_and_datasets_formats(self):
        raw = [["Evidence A", 0], ["Evidence B", 2]]
        datasets_format = {"title": ["Evidence A", "Evidence B"], "sent_id": [0, 2]}

        self.assertEqual(supporting_fact_titles(raw), {"Evidence A", "Evidence B"})
        self.assertEqual(supporting_fact_titles(datasets_format), {"Evidence A", "Evidence B"})

    def test_loader_uses_fully_qualified_hub_repository(self):
        calls = []
        fake_datasets = types.ModuleType("datasets")

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return "loaded"

        fake_datasets.load_dataset = fake_load_dataset
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            self.assertEqual(_load_hotpotqa_split("train"), "loaded")

        self.assertEqual(calls, [((HOTPOTQA_REPO_ID, "distractor"), {"split": "train"})])

    def test_select_samples_is_deterministic_and_checks_size(self):
        source = [sample(index) for index in range(10)]
        first = select_samples(source, count=4, seed=17)
        second = select_samples(source, count=4, seed=17)

        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        with self.assertRaises(ValueError):
            select_samples(source, count=11, seed=17)

    def test_partition_is_a_disjoint_70_30_split(self):
        train, evaluation = partition_samples([sample(index) for index in range(10)], 7, 3)

        self.assertEqual(len(train), 7)
        self.assertEqual(len(evaluation), 3)
        self.assertTrue({item["id"] for item in train}.isdisjoint(item["id"] for item in evaluation))

    def test_build_split_creates_isolated_train_and_eval_corpora(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "hotpotqa_mini"
            stats = build_split(
                [sample(index) for index in range(10)],
                output_dir,
                train_count=7,
                eval_count=3,
            )

            self.assertEqual(stats["train_tasks_count"], 7)
            self.assertEqual(stats["eval_tasks_count"], 3)
            self.assertEqual(stats["train_fraction"], 0.7)
            self.assertEqual(stats["eval_fraction"], 0.3)
            self.assertTrue(stats["overlap_verification"]["corpus_isolated_by_split"])

            train_task_paths = sorted((output_dir / "tasks" / "train").glob("*.json"))
            eval_task_paths = sorted((output_dir / "tasks" / "eval").glob("*.json"))
            self.assertEqual(len(train_task_paths), 7)
            self.assertEqual(len(eval_task_paths), 3)

            train_task = json.loads(train_task_paths[0].read_text(encoding="utf-8"))
            eval_task = json.loads(eval_task_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(train_task["reference_docs"], [train_task["task_id"]])
            self.assertEqual(eval_task["reference_docs"], [eval_task["task_id"]])
            train_corpus_doc = json.loads(
                (output_dir / "corpus" / "train" / f"{train_task['task_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            train_citation_ids = {chunk["chunk_id"] for chunk in train_corpus_doc["chunks"]}
            for citation in train_task["ground_truth_citations"]:
                self.assertIn(citation, train_citation_ids)
                self.assertFalse((output_dir / "corpus" / "eval" / f"{train_task['task_id']}.json").exists())

            train_chunk = json.loads(
                next((output_dir / "corpus" / "train").glob("*.json")).read_text(encoding="utf-8")
            )
            self.assertIn(train_chunk["doc_id"], {json.loads(path.read_text())["task_id"] for path in train_task_paths})
            self.assertEqual(len(list((output_dir / "corpus" / "train").glob("*.json"))), 7)

            corpus = CorpusStore(str(output_dir / "corpus" / "train"))
            corpus.load()
            results = corpus.search("evidence", doc_ids=train_task["reference_docs"])
            self.assertTrue(results)
            self.assertTrue(all(result.doc_id == train_task["task_id"] for result in results))

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "hotpotqa_debug"
            samples = [sample(index) for index in range(10)]
            build_split(samples, output_dir, train_count=7, eval_count=3)
            with self.assertRaises(FileExistsError):
                build_split(samples, output_dir, train_count=7, eval_count=3)
            stats = build_split(samples, output_dir, train_count=7, eval_count=3, overwrite=True)
            self.assertEqual(stats["total_tasks"], 10)

    def test_official_source_provenance_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            stats = build_benchmark(
                [sample(index) for index in range(7)],
                [sample(index + 7) for index in range(3)],
                Path(temporary_dir) / "hotpotqa_7k3k",
                source_splits={"train": "train", "eval": "validation"},
            )

            self.assertEqual(stats["source"]["dataset"], "hotpot_qa")
            self.assertEqual(stats["source"]["splits"], {"train": "train", "eval": "validation"})


if __name__ == "__main__":
    unittest.main()
