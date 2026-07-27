"""
tests/test_env.py
职责: Environment 核心功能测试
修复验证:
  1. corpus 注入: reset() 后 state._corpus 不为 None
  2. done_reason: step() 后 state.done_reason 正确设置
"""
from __future__ import annotations
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.env import ResearchEnv
from src.environment.corpus import CorpusStore
from src.environment.state import EnvState
from src.tools.search import SearchTool
from src.tools.read import ReadTool
from src.tools.rerank import RerankTool
from src.tools.cite import CiteTool
from src.tools.answer import AnswerTool
from src.schema.task import TaskSample, Rubric, TaskType
from src.schema.action import Action


def create_test_corpus() -> CorpusStore:
    """创建测试用 corpus。"""
    corpus = CorpusStore()
    from src.schema.document import Chunk, Document

    doc1 = Document(doc_id="doc1", title="Test Doc 1")
    chunk1 = Chunk(
        chunk_id="doc1_c1",
        doc_id="doc1",
        content="Reinforcement learning is a method for training AI models.",
        char_start=0,
        char_end=60,
        title="Test Doc 1",
    )
    doc1.add_chunk(chunk1)
    corpus.docs["doc1"] = doc1
    corpus.chunks["doc1_c1"] = chunk1
    corpus._chunk_ids.add("doc1_c1")
    corpus._doc_ids.add("doc1")
    return corpus


def create_test_task() -> TaskSample:
    """创建测试用 task。"""
    return TaskSample(
        task_id="test_task_001",
        task_type=TaskType.SURVEY_SYNTHESIS,
        user_query="What is reinforcement learning?",
        rubric=Rubric(),
        ground_truth_citations=["doc1_c1"],
        reference_docs=["doc1"],
    )


class TestEnvCorpusInjection(unittest.TestCase):
    """测试 corpus 注入问题修复。"""

    def test_reset_injects_corpus(self):
        """验证 reset() 将 corpus 注入到 state。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())

        obs = env.reset(task)

        # 修复验证: state._corpus 不为 None
        self.assertIsNotNone(env._state)
        self.assertIsNotNone(env._state._corpus)
        self.assertEqual(env._state._corpus, corpus)

    def test_tools_can_access_corpus_via_state(self):
        """验证工具可通过 state.get_corpus() 访问 corpus。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())

        env.reset(task)

        # 工具执行时能正确获取 corpus
        action = Action.search(query="reinforcement", topk=5)
        obs, done, reason = env.step(action)

        self.assertFalse(done)
        self.assertEqual(len(obs.candidate_chunks), 1)
        self.assertEqual(obs.candidate_chunks[0].chunk_id, "doc1_c1")


class TestEnvDoneReason(unittest.TestCase):
    """测试 done_reason 记录问题修复。"""

    def test_done_reason_set_on_max_steps(self):
        """验证达到最大步数时正确设置 done_reason。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=2)
        env.register_tool(SearchTool())
        env.register_tool(ReadTool())

        env.reset(task)

        # Step 1
        action1 = Action.search(query="reinforcement", topk=5)
        env.step(action1)

        # Step 2
        action2 = Action.search(query="learning", topk=5)
        obs, done, reason = env.step(action2)

        self.assertTrue(done)
        self.assertEqual(reason, "max_steps")
        # 修复验证: state.done_reason 正确设置
        self.assertEqual(env._state.done_reason, "max_steps")

    def test_done_reason_set_on_answer_submit(self):
        """验证提交答案时正确设置 done_reason。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())
        env.register_tool(ReadTool())
        env.register_tool(AnswerTool())

        env.reset(task)

        # 先搜索和读取
        env.step(Action.search(query="reinforcement"))
        env.step(Action.read(chunk_ids=["doc1_c1"]))

        # 提交答案
        obs, done, reason = env.step(Action.answer(
            answer_text="Reinforcement learning is a method.",
            cited_chunk_ids=["doc1_c1"]
        ))

        self.assertTrue(done)
        self.assertEqual(reason, "answer_submitted")
        # 修复验证: state.done_reason 正确设置
        self.assertEqual(env._state.done_reason, "answer_submitted")


class TestEnvStateTransitions(unittest.TestCase):
    """测试 Environment 状态转换。"""

    def test_full_episode_transitions(self):
        """测试完整 episode 的状态转换。"""
        corpus = create_test_corpus()
        task = create_test_task()
        env = ResearchEnv(corpus=corpus, max_steps=10)
        env.register_tool(SearchTool())
        env.register_tool(ReadTool())
        env.register_tool(CiteTool())
        env.register_tool(AnswerTool())

        env.reset(task)

        # SEARCH
        obs1, done1, _ = env.step(Action.search(query="reinforcement"))
        self.assertFalse(done1)
        self.assertEqual(len(obs1.candidate_chunks), 1)
        self.assertEqual(obs1.remaining_steps, 9)

        # READ
        obs2, done2, _ = env.step(Action.read(chunk_ids=["doc1_c1"]))
        self.assertFalse(done2)
        self.assertEqual(len(obs2.read_summaries), 1)
        self.assertEqual(obs2.remaining_steps, 8)

        # CITE
        obs3, done3, _ = env.step(Action.cite(
            chunk_ids=["doc1_c1"],
            claims=["Reinforcement learning is mentioned"]
        ))
        self.assertFalse(done3)
        self.assertEqual(len(obs3.cited_chunks), 1)
        self.assertEqual(obs3.remaining_steps, 7)

        # ANSWER
        obs4, done4, reason = env.step(Action.answer(
            answer_text="Reinforcement learning is a method.",
            cited_chunk_ids=["doc1_c1"]
        ))
        self.assertTrue(done4)
        self.assertEqual(reason, "answer_submitted")

        # Finalize
        result = env.finalize_episode()
        self.assertEqual(result.done_reason, "answer_submitted")
        self.assertEqual(result.final_answer, "Reinforcement learning is a method.")
        self.assertEqual(result.cited_chunk_ids, ["doc1_c1"])


if __name__ == "__main__":
    unittest.main()
