from __future__ import annotations
import unittest
from research_agent.core.env.env import ResearchEnv
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.schema.task import TaskSample, Rubric, TaskType
from research_agent.core.schema.action import Action
from research_agent.core.tools.search import SearchTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.answer import AnswerTool
from research_agent.core.tools.cite import CiteTool

class TestEnvStepFlow(unittest.TestCase):
    def setUp(self):
        # Create a tiny corpus store
        self.corpus = CorpusStore()
        from research_agent.core.schema.document import Chunk, Document
        doc = Document(doc_id="doc1", title="Test Doc")
        chunk = Chunk(
            chunk_id="doc1_c1",
            doc_id="doc1",
            content="Water is H2O.",
            char_start=0,
            char_end=12,
            title="Test Doc"
        )
        doc.add_chunk(chunk)
        self.corpus.docs["doc1"] = doc
        self.corpus.chunks["doc1_c1"] = chunk
        self.corpus._chunk_ids.add("doc1_c1")
        self.corpus._doc_ids.add("doc1")

        # Create task sample
        self.task = TaskSample(
            task_id="t1",
            task_type=TaskType.SURVEY_SYNTHESIS,
            user_query="What is water?",
            rubric=Rubric(),
            ground_truth_answer="H2O",
            ground_truth_citations=["doc1_c1"],
            reference_docs=["doc1"]
        )

        # Setup Env
        self.env = ResearchEnv(corpus=self.corpus, max_steps=5)
        self.env.register_tool(SearchTool())
        self.env.register_tool(ReadTool())
        self.env.register_tool(CiteTool())
        self.env.register_tool(AnswerTool())

    def test_reset_and_step_sequence(self):
        # Reset
        obs = self.env.reset(self.task)
        self.assertEqual(obs.remaining_steps, 5)
        self.assertEqual(len(obs.candidate_chunks), 0)

        # SEARCH step
        obs, done, reason = self.env.step(Action.search(query="water"))
        self.assertFalse(done)
        self.assertEqual(len(obs.candidate_chunks), 1)
        self.assertEqual(obs.candidate_chunks[0].chunk_id, "doc1_c1")
        self.assertEqual(obs.remaining_steps, 4)

        # READ step
        obs, done, reason = self.env.step(Action.read(chunk_ids=["doc1_c1"]))
        self.assertFalse(done)
        self.assertEqual(len(obs.read_summaries), 1)
        self.assertEqual(obs.read_summaries[0].chunk_id, "doc1_c1")
        self.assertEqual(obs.remaining_steps, 3)

        # CITE step
        obs, done, reason = self.env.step(Action.cite(["doc1_c1"], ["Water is H2O."]))
        self.assertFalse(done)
        self.assertEqual(obs.cited_chunks, ["doc1_c1"])

        # ANSWER step
        obs, done, reason = self.env.step(Action.answer(answer_text="Water is H2O.", cited_chunk_ids=["doc1_c1"]))
        self.assertTrue(done)
        self.assertEqual(reason, "answer_submitted")
        self.assertEqual(obs.final_answer, "Water is H2O.")

        # Finalize
        result = self.env.finalize_episode()
        self.assertEqual(result.done_reason, "answer_submitted")
        self.assertEqual(result.final_answer, "Water is H2O.")
        self.assertEqual(result.cited_chunk_ids, ["doc1_c1"])
        self.assertEqual(result.total_steps, 4)

    def test_invalid_action_penalty(self):
        self.env.reset(self.task)
        # Try invalid action
        obs, done, reason = self.env.step(Action(tool="INVALID", intent="bad", params={}))
        self.assertFalse(done)
        self.assertEqual(obs.invalid_action_count, 1)
        self.assertEqual(obs.remaining_steps, 4)

        # Try another invalid action
        obs, done, reason = self.env.step(Action(tool="INVALID", intent="bad", params={}))
        self.assertFalse(done)
        self.assertEqual(obs.invalid_action_count, 2)

        # Third invalid action should terminate the episode (max_invalid_actions = 3)
        obs, done, reason = self.env.step(Action(tool="INVALID", intent="bad", params={}))
        self.assertTrue(done)
        self.assertEqual(reason, "invalid_actions_exceeded")

    def test_parser_error_message_is_exposed_to_next_observation(self):
        self.env.reset(self.task)
        self.env.step(Action(tool="INVALID", intent="Action 'params' must be an object", params={}))

        self.assertEqual(
            self.env._state.trajectory[-1].error_message,
            "Action 'params' must be an object",
        )

if __name__ == "__main__":
    unittest.main()
