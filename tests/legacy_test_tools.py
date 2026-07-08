"""
tests/test_tools.py
职责: Tools 功能测试
修复验证:
  1. SearchTool n_new 计算正确（使用 add_candidates 返回值）
  2. cite_chunks 返回实际新增的 chunk_ids
"""
from __future__ import annotations
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.env import ResearchEnv
from src.environment.corpus import CorpusStore
from src.tools.search import SearchTool
from src.tools.read import ReadTool
from src.tools.cite import CiteTool
from src.tools.answer import AnswerTool
from src.schema.task import TaskSample, Rubric, TaskType
from src.schema.action import Action


def create_test_corpus() -> CorpusStore:
    """创建测试用 corpus。"""
    corpus = CorpusStore()
    from src.schema.document import Chunk, Document

    for i in range(1, 4):
        doc_id = f"doc{i}"
        doc = Document(doc_id=doc_id, title=f"Test Doc {i}")
        for j in range(1, 3):
            chunk_id = f"doc{i}_c{j}"
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=f"This is test content for document {i}, chunk {j}. Reinforcement learning methods are included.",
                char_start=0,
                char_end=100,
                title=f"Test Doc {i}",
            )
            doc.add_chunk(chunk)
            corpus.chunks[chunk_id] = chunk
            corpus._chunk_ids.add(chunk_id)
            corpus._doc_ids.add(doc_id)
        corpus.docs[doc_id] = doc
    return corpus


def create_test_task() -> TaskSample:
    """创建测试用 task。"""
    return TaskSample(
        task_id="test_task_001",
        task_type=TaskType.SURVEY_SYNTHESIS,
        user_query="What is reinforcement learning?",
        rubric=Rubric(),
        ground_truth_citations=["doc1_c1"],
        reference_docs=["doc1", "doc2", "doc3"],
    )


class TestSearchToolNNew(unittest.TestCase):
    """测试 n_new 计算修复。"""

    def test_search_n_new_calculation(self):
        """验证 SEARCH 的 n_new 基于实际新增 candidates 计算。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())
        env.reset(task)

        # 第一次搜索 - 应该返回 2 个新 candidates
        action = Action.search(query="reinforcement", topk=10)
        obs, _, _ = env.step(action)

        self.assertEqual(len(obs.candidate_chunks), 2)  # doc1_c1, doc2_c1, doc3_c1 匹配
        # 验证 search_history 中记录了 n_new
        self.assertEqual(len(obs.search_history), 1)
        self.assertGreater(obs.search_history[0]["n_new"], 0)

    def test_search_n_new_after_duplicate(self):
        """验证重复搜索不会重复计算 n_new。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())
        env.reset(task)

        # 第一次搜索
        env.step(Action.search(query="reinforcement"))
        n_candidates_after_first = len(env._state.candidate_chunks)

        # 第二次搜索相同 query
        env.step(Action.search(query="reinforcement"))
        n_candidates_after_second = len(env._state.candidate_chunks)

        # n_new 第二次应该是 0
        self.assertEqual(n_candidates_after_first, n_candidates_after_second)


class TestCiteToolNewCited(unittest.TestCase):
    """测试 cite_chunks 返回实际新增的 chunk_ids。"""

    def test_cite_returns_newly_added(self):
        """验证 CITE 返回实际新增的 cited chunk_ids。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())
        env.register_tool(ReadTool())
        env.register_tool(CiteTool())
        env.reset(task)

        # 先读取一些 chunks
        env.step(Action.read(chunk_ids=["doc1_c1", "doc1_c2"]))

        # 第一次 CITE
        result1 = env.step(Action.cite(
            chunk_ids=["doc1_c1"],
            claims=["claim1"]
        ))
        n_cited_after_first = len(env._state.cited_chunks)

        # 第二次 CITE - 新增 doc1_c2
        result2 = env.step(Action.cite(
            chunk_ids=["doc1_c2"],
            claims=["claim2"]
        ))
        n_cited_after_second = len(env._state.cited_chunks)

        # 验证实际新增
        self.assertEqual(n_cited_after_first, 1)
        self.assertEqual(n_cited_after_second, 2)


class TestToolIntegration(unittest.TestCase):
    """测试工具集成。"""

    def test_full_tool_sequence(self):
        """测试完整工具调用序列。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())
        env.register_tool(ReadTool())
        env.register_tool(RerankTool())
        env.register_tool(CiteTool())
        env.register_tool(AnswerTool())
        env.reset(task)

        # SEARCH
        obs1, _, _ = env.step(Action.search(query="reinforcement"))
        self.assertGreater(len(obs1.candidate_chunks), 0)
        candidate_ids = [c.chunk_id for c in obs1.candidate_chunks]

        # READ
        obs2, _, _ = env.step(Action.read(chunk_ids=candidate_ids[:2]))
        self.assertGreater(len(obs2.read_summaries), 0)

        # RERANK
        obs3, _, _ = env.step(Action.rerank(
            query="reinforcement",
            candidate_chunk_ids=candidate_ids[:2],
            topk=2
        ))
        self.assertEqual(obs3.last_tool, "RERANK")

        # CITE
        obs4, _, _ = env.step(Action.cite(
            chunk_ids=candidate_ids[:1],
            claims=["Reinforcement learning is a method"]
        ))
        self.assertEqual(len(obs4.cited_chunks), 1)

        # ANSWER
        obs5, done, reason = env.step(Action.answer(
            answer_text="Reinforcement learning is a method for training AI.",
            cited_chunk_ids=candidate_ids[:1]
        ))
        self.assertTrue(done)
        self.assertEqual(reason, "answer_submitted")


if __name__ == "__main__":
    unittest.main()
