from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research_agent.core.effect_space import (
    SearchProposal,
    effect_key,
    finite_effect_kl_decomposition,
    group_by_effect,
    jaccard,
    shared_effect_advantages,
)
from research_agent.core.schema.action import Action
from scripts.effect_space_pilot import analyze_samples, sample_first_search
from scripts.prepare_musique import build_benchmark
from tests.test_musique_preparation import example


class TestEffectSpace(unittest.TestCase):
    def test_effect_kl_chain_rule_exposes_within_class_drift(self):
        effects = {"a1": ("chunk_a",), "a2": ("chunk_a",), "b": ("chunk_b",)}
        full, between, within = finite_effect_kl_decomposition(
            {"a1": 0.4, "a2": 0.1, "b": 0.5},
            {"a1": 0.2, "a2": 0.3, "b": 0.5},
            effects,
        )
        self.assertAlmostEqual(full, between + within)
        self.assertAlmostEqual(between, 0.0)
        self.assertGreater(within, 0.0)
        # Identical retrieved IDs do not imply identical value when the raw
        # query remains in the next prompt and can alter continuation quality.
        q = {"a1": 0.0, "a2": 1.0, "b": 0.0}
        old_value = sum(prob * q[action] for action, prob in
                        {"a1": 0.4, "a2": 0.1, "b": 0.5}.items())
        new_value = sum(prob * q[action] for action, prob in
                        {"a1": 0.2, "a2": 0.3, "b": 0.5}.items())
        self.assertGreater(new_value, old_value)

    def test_effect_kl_requires_complete_finite_distributions(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            finite_effect_kl_decomposition(
                {"a": 0.2}, {"a": 1.0}, {"a": ("chunk",)}
            )
        with self.assertRaisesRegex(ValueError, "absolutely continuous"):
            finite_effect_kl_decomposition(
                {"a": 0.0, "b": 1.0}, {"a": 0.5, "b": 0.5},
                {"a": ("chunk",), "b": ("other",)},
            )

    def test_exact_ordered_effect_and_shared_advantage(self):
        a = effect_key(["first", "second"])
        b = effect_key(["second", "first"])
        self.assertNotEqual(a, b)
        self.assertEqual(jaccard(a, b), 1.0)
        proposals = [
            SearchProposal("first query", a),
            SearchProposal("different words", a),
            SearchProposal("other effect", b),
        ]
        self.assertEqual(group_by_effect(proposals), {a: [0, 1], b: [2]})
        self.assertEqual(
            shared_effect_advantages(proposals, {a: [1.0, 0.0], b: [0.0]},
                                     state_baseline=0.25),
            [0.25, 0.25, -0.25],
        )

    def test_invalid_effects_or_missing_continuation_are_rejected(self):
        with self.assertRaises(ValueError):
            effect_key(["same", "same"])
        with self.assertRaises(ValueError):
            effect_key([""])
        with self.assertRaisesRegex(ValueError, "missing continuation"):
            shared_effect_advantages([SearchProposal("q", ("a",))], {},
                                     state_baseline=0.0)

    def test_pilot_detects_distinct_queries_with_same_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev = root / "train.jsonl", root / "dev.jsonl"
            train.write_text(json.dumps(example("2hop__1_2", "Ada")) + "\n", encoding="utf-8")
            dev.write_text(json.dumps(example("2hop__3_4", "Bela")) + "\n", encoding="utf-8")
            benchmark = root / "benchmark"
            build_benchmark(train, dev, benchmark)
            task = json.loads(next((benchmark / "tasks" / "train").glob("*.json")).read_text())
            rows = [
                {"task_id": task["task_id"], "sample_index": 0,
                 "valid_search": True, "query": "Ada founded", "topk": 2},
                {"task_id": task["task_id"], "sample_index": 1,
                 "valid_search": True, "query": "Ada founded?", "topk": 2},
                {"task_id": task["task_id"], "sample_index": 2,
                 "valid_search": False, "query": None},
            ]
            summary, analyzed = analyze_samples(
                rows, benchmark / "tasks" / "train", benchmark / "corpus" / "train"
            )
            self.assertEqual(summary["valid_searches"], 2)
            self.assertEqual(summary["unique_queries"], 2)
            self.assertEqual(summary["unique_ordered_effects"], 1)
            self.assertEqual(summary["distinct_query_effect_collisions"], 1)
            self.assertEqual(summary["valid_search_rate"], 2 / 3)
            self.assertEqual(analyzed[0]["effect_chunk_ids"], analyzed[1]["effect_chunk_ids"])

    def test_same_ids_with_changed_scores_are_reported_as_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            (tasks / "task_1.json").write_text(json.dumps({
                "task_id": "task_1", "ground_truth_citations": ["c1"],
                "reference_docs": [],
            }), encoding="utf-8")
            rows = [
                {"task_id": "task_1", "sample_index": 0,
                 "valid_search": True, "query": "query a", "topk": 1},
                {"task_id": "task_1", "sample_index": 1,
                 "valid_search": True, "query": "query b", "topk": 1},
            ]

            class FakeCorpus:
                def __init__(self, path):
                    self.path = path

                def load(self):
                    pass

                def close(self):
                    pass

                def __len__(self):
                    return 1

                def search(self, query, topk, doc_ids):
                    return [SimpleNamespace(chunk_id="c1", score=1.0 if query == "query a" else 2.0)]

            with patch("scripts.effect_space_pilot.CorpusStore", FakeCorpus):
                summary, analyzed = analyze_samples(rows, tasks, root)
            self.assertEqual(summary["distinct_query_effect_collisions"], 1)
            self.assertEqual(summary["same_ids_classes_with_score_shift"], 1)
            self.assertEqual(summary["max_same_ids_score_shift"], 1.0)
            self.assertEqual(analyzed[0]["effect_chunk_ids"], analyzed[1]["effect_chunk_ids"])

    def test_sampling_resume_keeps_existing_rows_and_rejects_config_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks_dir = root / "tasks"
            tasks_dir.mkdir()
            (tasks_dir / "task_1.json").write_text(json.dumps({
                "task_id": "task_1", "user_query": "Who founded the Comets?",
                "retrieval_scope": "split_corpus",
            }), encoding="utf-8")
            sample_path = root / "samples.jsonl"

            class FakeTurn:
                action = Action.search("Comets founder", topk=3)
                raw_content = "<action>...</action>"
                prompt_tokens = 10
                completion_tokens = 4

            kwargs = {
                "max_tasks": 1, "samples_per_task": 2, "selection_seed": 42,
                "temperature": 1.0, "top_p": 0.95, "max_tokens": 256,
                "sample_path": sample_path, "model_revision": "checkpoint-test-1",
            }
            with patch("scripts.effect_space_pilot.wait_for_model_server"), patch(
                "scripts.effect_space_pilot.LLMActor"
            ) as actor_cls:
                actor_cls.return_value.decide.return_value = FakeTurn()
                first = sample_first_search(tasks_dir, "http://127.0.0.1:8000/v1", "model",
                                            **kwargs)
                second = sample_first_search(tasks_dir, "http://127.0.0.1:8000/v1", "model",
                                             resume=True, **kwargs)
                self.assertEqual(actor_cls.return_value.decide.call_count, 2)
                self.assertEqual(first, second)
                self.assertEqual(len(sample_path.read_text().splitlines()), 2)
                self.assertEqual(
                    json.loads((root / "sampling_config.json").read_text())["model_revision"],
                    "checkpoint-test-1",
                )
                with self.assertRaisesRegex(ValueError, "do not match"):
                    sample_first_search(tasks_dir, "http://127.0.0.1:8000/v1", "model",
                                        resume=True, **{**kwargs, "temperature": 0.7})
                with self.assertRaisesRegex(ValueError, "do not match"):
                    sample_first_search(tasks_dir, "http://127.0.0.1:8000/v1", "model",
                                        resume=True,
                                        **{**kwargs, "model_revision": "checkpoint-test-2"})


if __name__ == "__main__":
    unittest.main()
