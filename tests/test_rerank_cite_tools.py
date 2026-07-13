from __future__ import annotations

import unittest

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.env import ResearchEnv
from research_agent.core.schema.action import Action
from research_agent.core.schema.document import Chunk, Document
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.cite import CiteTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.rerank import RerankTool
from research_agent.core.tools.search import SearchTool


class TestRerankAndCiteTools(unittest.TestCase):
    def setUp(self):
        corpus = CorpusStore()
        document = Document(doc_id="doc", title="Evidence")
        for chunk_id, content in [
            ("weak", "Canada has many lakes."),
            ("strong", "The flag of Canada has a red maple leaf."),
        ]:
            chunk = Chunk(chunk_id, "doc", content, 0, len(content), "Evidence")
            document.add_chunk(chunk)
            corpus.chunks[chunk_id] = chunk
            corpus._chunk_ids.add(chunk_id)
        corpus.docs["doc"] = document
        corpus._doc_ids.add("doc")

        self.env = ResearchEnv(corpus=corpus, max_steps=6)
        for tool in [SearchTool(), ReadTool(), RerankTool(), CiteTool()]:
            self.env.register_tool(tool)
        self.task = TaskSample(
            task_id="canada",
            task_type=TaskType.SURVEY_SYNTHESIS,
            user_query="Which flag has a maple leaf?",
            rubric=Rubric(),
        )

    def test_rerank_then_cite_read_evidence(self):
        self.env.reset(self.task)
        self.env.step(Action.search("Canada flag maple leaf", topk=2))
        observation, done, _ = self.env.step(
            Action.rerank("flag maple leaf", ["weak", "strong"], topk=1)
        )
        self.assertFalse(done)
        self.assertEqual(observation.candidate_chunks[0].chunk_id, "strong")

        self.env.step(Action.read(["strong"]))
        observation, done, _ = self.env.step(
            Action.cite(["strong"], ["Canada's flag has a red maple leaf."])
        )
        self.assertFalse(done)
        self.assertEqual(observation.cited_chunks, ["strong"])

    def test_cite_rejects_evidence_that_was_not_read(self):
        self.env.reset(self.task)
        observation, done, _ = self.env.step(Action.cite(["strong"], ["unsupported claim"]))
        self.assertFalse(done)
        self.assertEqual(observation.invalid_action_count, 1)


if __name__ == "__main__":
    unittest.main()
