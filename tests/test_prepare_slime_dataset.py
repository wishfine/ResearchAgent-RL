from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_slime_dataset import build_records, write_dataset


def task(index: int) -> dict:
    return {
        "task_id": f"hotpot_{index}",
        "user_query": f"Question {index}?",
        "ground_truth_answer": f"Answer {index}",
        "ground_truth_citations": [f"chunk_{index}"],
        "reference_docs": [f"hotpot_{index}"],
    }


class TestPrepareSlimeDataset(unittest.TestCase):
    def test_writes_metadata_preserving_jsonl_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tasks_dir = root / "tasks"
            tasks_dir.mkdir()
            for index in range(3):
                (tasks_dir / f"task_{index}.json").write_text(
                    json.dumps(task(index)), encoding="utf-8"
                )

            records, manifest = build_records(tasks_dir, max_samples=2, selection_seed=7)
            output_path, manifest_path = write_dataset(records, manifest, root / "train.jsonl")

            output_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(output_records), 2)
            self.assertEqual(output_records[0]["label"], output_records[0]["metadata"]["ground_truth_answer"])
            self.assertEqual(
                output_records[0]["prompt"],
                [{"role": "user", "content": output_records[0]["metadata"]["user_query"]}],
            )
            self.assertTrue(output_records[0]["metadata"]["reference_docs"])
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["records"], 2)
            self.assertEqual(saved_manifest["selection_seed"], 7)
            self.assertEqual(len(saved_manifest["records_sha256"]), 64)

    def test_rejects_task_without_retrieval_scope(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            tasks_dir = Path(temporary_dir) / "tasks"
            tasks_dir.mkdir()
            invalid = task(0)
            invalid["reference_docs"] = []
            (tasks_dir / "invalid.json").write_text(json.dumps(invalid), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reference_docs"):
                build_records(tasks_dir)


if __name__ == "__main__":
    unittest.main()
