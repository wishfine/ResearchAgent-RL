from __future__ import annotations

import unittest

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.state import EnvState
from research_agent.core.schema.document import Chunk, Document
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.search import SearchTool


class TestSearchTool(unittest.TestCase):
    def test_scoped_complete_search_includes_zero_score_candidates(self):
        corpus = CorpusStore()
        document = Document(doc_id="task", title="task")
        matching = Chunk("matching", "task", "maple leaf", 0, 10, "Matching")
        supporting = Chunk("supporting", "task", "unrelated evidence", 0, 18, "Supporting")
        for chunk in (matching, supporting):
            document.add_chunk(chunk)
            corpus.chunks[chunk.chunk_id] = chunk
            corpus._chunk_ids.add(chunk.chunk_id)
        corpus.docs[document.doc_id] = document
        corpus._doc_ids.add(document.doc_id)

        task = TaskSample(
            task_id="task",
            task_type=TaskType.SURVEY_SYNTHESIS,
            user_query="maple leaf",
            rubric=Rubric(),
            reference_docs=["task"],
        )
        state = EnvState(task=task, _corpus=corpus)

        result = SearchTool().execute({"query": "maple leaf", "topk": 2}, state)

        self.assertTrue(result.success)
        self.assertEqual(
            {candidate["chunk_id"] for candidate in result.data["candidates"]},
            {"matching", "supporting"},
        )


if __name__ == "__main__":
    unittest.main()
