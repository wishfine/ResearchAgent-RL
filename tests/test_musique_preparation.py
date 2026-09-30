from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.env import ResearchEnv
from research_agent.core.env.state import EnvState
from research_agent.core.schema.action import Action
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.answer import AnswerTool
from research_agent.core.tools.cite import CiteTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.search import SearchTool
from scripts.evaluate_retrieval import evaluate
from scripts.prepare_musique import build_benchmark
from scripts.prepare_slime_dataset import build_records


def example(identifier: str, entity: str) -> dict:
    return {
        "id": identifier, "answerable": True,
        "question": f"Who coached the team founded by {entity}?",
        "answer": "Morgan", "answer_aliases": [],
        "paragraphs": [
            {"idx": 0, "title": entity, "paragraph_text": f"{entity} founded the Comets team.",
             "is_supporting": True},
            {"idx": 1, "title": "Comets team", "paragraph_text": f"The Comets team founded by {entity} was coached by Morgan.",
             "is_supporting": True},
            {"idx": 2, "title": "Unrelated club", "paragraph_text": "A different club won in 1999.",
             "is_supporting": False},
        ],
        "question_decomposition": [
            {"id": 1, "question": f"What team did {entity} found?", "answer": "Comets team",
             "paragraph_support_idx": 0},
            {"id": 2, "question": "Who coached #1?", "answer": "Morgan",
             "paragraph_support_idx": 1},
        ],
    }


class TestMuSiQuePreparation(unittest.TestCase):
    def test_pooled_search_and_rl_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_file, dev_file = root / "train.jsonl", root / "dev.jsonl"
            training = example("2hop__1_2", "Ada")
            leaked_eval_gold = dict(example("2hop__3_4", "Bela")["paragraphs"][0])
            leaked_eval_gold.update(idx=3, is_supporting=False)
            training["paragraphs"].append(leaked_eval_gold)
            train_file.write_text(json.dumps(training) + "\n", encoding="utf-8")
            dev_file.write_text(json.dumps(example("2hop__3_4", "Bela")) + "\n", encoding="utf-8")
            output = root / "musique_rl"
            stats = build_benchmark(train_file, dev_file, output)

            self.assertEqual((stats["train_tasks_count"], stats["eval_tasks_count"]), (1, 1))
            self.assertEqual(stats["task_id_overlap_count"], 0)
            self.assertEqual(stats["eval_gold_excluded_from_train_corpus"], 1)
            task_path = next((output / "tasks" / "train").glob("*.json"))
            task = json.loads(task_path.read_text(encoding="utf-8"))
            self.assertEqual(task["retrieval_scope"], "split_corpus")
            self.assertEqual(len(task["ground_truth_citations"]), 2)

            corpus = CorpusStore(str(output / "corpus" / "train"))
            corpus.load()
            self.assertEqual(len(corpus), 3)
            first = corpus.search("Ada founded", topk=2)
            self.assertEqual(first[0].title, "Ada")
            second = corpus.search("Comets coached Morgan", topk=2)
            self.assertEqual(second[0].title, "Comets team")
            self.assertTrue(all(candidate.chunk_id in corpus for candidate in second))
            self.assertTrue(corpus.search('Ada" OR *', topk=2))
            self.assertEqual(corpus.search("no_matching_token_123", topk=2), [])

            runtime_task = TaskSample(
                task_id=task["task_id"], task_type=TaskType.SURVEY_SYNTHESIS,
                user_query=task["user_query"], rubric=Rubric(),
                ground_truth_citations=task["ground_truth_citations"],
                retrieval_scope="split_corpus",
            )
            state = EnvState(task=runtime_task, _corpus=corpus)
            search_result = SearchTool().execute({"query": "Ada founded", "topk": 2}, state)
            self.assertTrue(search_result.success)
            read_result = ReadTool().execute(
                {"chunk_ids": [search_result.data["candidates"][0]["chunk_id"]]}, state
            )
            self.assertTrue(read_result.success)

            env = ResearchEnv(corpus, max_steps=6)
            for tool in (SearchTool(), ReadTool(), CiteTool(), AnswerTool()):
                env.register_tool(tool)
            env.reset(runtime_task)
            first_id, second_id = task["ground_truth_citations"]
            actions = [
                Action.search("Ada founded", topk=2), Action.read([first_id]),
                Action.search("Comets coached Morgan", topk=2), Action.read([second_id]),
                Action.cite([first_id, second_id], ["Ada founded Comets; Morgan coached them."]),
                Action.answer("Morgan", [first_id, second_id]),
            ]
            for action in actions:
                _, done, _ = env.step(action)
            self.assertTrue(done)
            self.assertEqual(env.finalize_episode().cited_chunk_ids, [first_id, second_id])

            records, _ = build_records(output / "tasks" / "train")
            self.assertEqual(records[0]["metadata"]["retrieval_scope"], "split_corpus")
            metrics = evaluate(output / "tasks" / "train", output / "corpus" / "train", [1, 3])
            self.assertEqual(metrics["single_search"]["3"]["all_support_recall"], 1.0)
            corpus.close()

    def test_rejects_unanswerable_or_missing_hop_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = example("2hop__1_2", "Ada")
            row["question_decomposition"][1]["paragraph_support_idx"] = 99
            source = root / "train.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dev = root / "dev.jsonl"
            dev.write_text(json.dumps(example("2hop__3_4", "Bela")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hop has no supporting paragraph"):
                build_benchmark(source, dev, root / "out")


if __name__ == "__main__":
    unittest.main()
