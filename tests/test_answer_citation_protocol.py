from __future__ import annotations

import unittest

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.env import ResearchEnv
from research_agent.core.schema.action import Action
from research_agent.core.schema.document import Chunk, Document
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.answer import AnswerTool
from research_agent.core.tools.cite import CiteTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.search import SearchTool


class TestAnswerCitationProtocol(unittest.TestCase):
    def setUp(self):
        self.corpus = CorpusStore()
        document = Document(doc_id="doc", title="Evidence")
        chunk = Chunk("evidence", "doc", "Canada has a maple leaf flag.", 0, 30)
        document.add_chunk(chunk)
        self.corpus.docs["doc"] = document
        self.corpus.chunks["evidence"] = chunk
        self.corpus._chunk_ids.add("evidence")
        self.corpus._doc_ids.add("doc")

    def _make_env(self):
        env = ResearchEnv(corpus=self.corpus, max_steps=5)
        for tool in [SearchTool(), ReadTool(), CiteTool(), AnswerTool()]:
            env.register_tool(tool)
        return env

    def test_answer_rejects_direct_citation_without_cite_action(self):
        env = self._make_env()
        task = TaskSample("task", TaskType.SURVEY_SYNTHESIS, "Which country?", Rubric())

        env.reset(task)
        env.step(Action.search("Canada"))
        env.step(Action.read(["evidence"]))
        observation, done, _ = env.step(Action.answer("Canada", ["evidence"]))

        self.assertFalse(done)
        self.assertEqual(observation.invalid_action_count, 1)
        self.assertIsNone(observation.final_answer)

    def test_answer_rejects_empty_citations(self):
        env = self._make_env()
        task = TaskSample("task", TaskType.SURVEY_SYNTHESIS, "Which country?", Rubric())

        env.reset(task)
        observation, done, _ = env.step(Action.answer("Canada", []))

        self.assertFalse(done)
        self.assertEqual(observation.invalid_action_count, 1)
        self.assertIsNone(observation.final_answer)


if __name__ == "__main__":
    unittest.main()
