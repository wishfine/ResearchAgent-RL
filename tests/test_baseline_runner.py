from __future__ import annotations

import unittest

from research_agent.baselines.llm_actor import ActorTurn
from research_agent.core.baseline_runner import aggregate_records, run_episode
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.env import ResearchEnv
from research_agent.core.schema.action import Action
from research_agent.core.schema.document import Chunk, Document
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.answer import AnswerTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.search import SearchTool


class _SequenceActor:
    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, prompt, **kwargs):
        action = next(self.actions)
        return ActorTurn(
            raw_content="<action>{}</action>",
            action=action,
            reasoning="kept separately",
            latency_sec=0.25,
            prompt_tokens=10,
            completion_tokens=5,
        )


class TestBaselineRunner(unittest.TestCase):
    def setUp(self):
        corpus = CorpusStore()
        document = Document(doc_id="doc1", title="Canada")
        chunk = Chunk(
            chunk_id="maple_leaf_flag",
            doc_id="doc1",
            content="Canada's flag features a red maple leaf.",
            char_start=0,
            char_end=40,
            title="Canada",
        )
        document.add_chunk(chunk)
        corpus.docs[document.doc_id] = document
        corpus.chunks[chunk.chunk_id] = chunk
        corpus._chunk_ids.add(chunk.chunk_id)
        corpus._doc_ids.add(document.doc_id)

        self.env = ResearchEnv(corpus=corpus, max_steps=4)
        self.env.register_tool(SearchTool())
        self.env.register_tool(ReadTool())
        self.env.register_tool(AnswerTool())
        self.task = TaskSample(
            task_id="canada",
            task_type=TaskType.SURVEY_SYNTHESIS,
            user_query="Which country has a red maple leaf flag?",
            rubric=Rubric(),
            ground_truth_answer="Canada",
            ground_truth_citations=["maple_leaf_flag"],
        )

    def test_records_full_three_step_episode_and_metrics(self):
        actor = _SequenceActor(
            [
                Action.search("Canada maple leaf", topk=1),
                Action.read(["maple_leaf_flag"]),
                Action.answer("Canada", ["maple_leaf_flag"]),
            ]
        )

        record = run_episode(self.env, self.task, actor, max_tokens=64)

        self.assertEqual(record["termination_reason"], "answer_submitted")
        self.assertEqual(record["final_answer"], "Canada")
        self.assertEqual(len(record["steps"]), 3)
        self.assertEqual(record["token_usage"], {"prompt_tokens": 30, "completion_tokens": 15})
        self.assertEqual(record["eval_metrics"]["citation_f1"], 1.0)
        self.assertTrue(record["eval_metrics"]["task_success"])
        self.assertEqual(record["steps"][0]["observation"]["task_id"], "canada")

    def test_aggregates_baseline_acceptance_metrics(self):
        record = {
            "eval_metrics": {
                "action_parse_success_rate": 1.0,
                "invalid_action_rate": 0.0,
                "task_success": True,
                "answer_quality": 1.0,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "citation_f1": 1.0,
                "average_steps": 3.0,
                "repeated_query_rate": 0.0,
            },
            "token_usage": {"prompt_tokens": 30, "completion_tokens": 15},
            "latency_sec": 0.75,
        }

        summary = aggregate_records([record])

        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["average_prompt_tokens"], 30.0)
        self.assertEqual(summary["average_episode_latency_sec"], 0.75)
        self.assertEqual(summary["citation_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
