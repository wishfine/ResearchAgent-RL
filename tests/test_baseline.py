"""
tests/test_baseline.py
职责: Rule-Based Baseline 测试
"""
from __future__ import annotations
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.rule_based import RuleBasedPolicy
from src.schema.observation import Observation
from src.schema.action import Action


def create_test_observation(
    n_candidates=0,
    n_read=0,
    n_cited=0,
    remaining_steps=10,
    no_progress=0,
    last_tool=None,
) -> Observation:
    """创建测试用 observation。"""
    return Observation(
        task_id="test_task",
        task_type="survey_synthesis",
        user_query="What is reinforcement learning?",
        current_subgoal="执行首次检索",
        current_evidence_gap="尚无候选文档",
        candidate_chunks=[],
        read_summaries=[],
        cited_chunks=[],
        search_history=[],
        failed_searches=[],
        trajectory=[],
        last_tool=last_tool,
        remaining_steps=remaining_steps,
        invalid_action_count=0,
        repeated_action_count=0,
        no_progress_count=no_progress,
        final_answer=None,
    )


class TestRuleBasedPolicyDecisions(unittest.TestCase):
    """测试 rule-based 策略决策。"""

    def test_decides_search_when_no_candidates(self):
        """验证无候选时决定 SEARCH。"""
        policy = RuleBasedPolicy()
        obs = create_test_observation(n_candidates=0, remaining_steps=10)

        action = policy.decide(obs)

        self.assertEqual(action.tool, "SEARCH")

    def test_decides_read_when_candidates_exist(self):
        """验证有候选时决定 READ。"""
        policy = RuleBasedPolicy()
        from src.schema.document import CandidateChunk
        obs = create_test_observation(
            n_candidates=2,
            n_read=0,
            remaining_steps=10,
        )
        obs.candidate_chunks = [
            CandidateChunk(chunk_id="c1", doc_id="d1", score=1.0, rank=1, query="test"),
            CandidateChunk(chunk_id="c2", doc_id="d1", score=0.8, rank=2, query="test"),
        ]

        action = policy.decide(obs)

        self.assertEqual(action.tool, "READ")

    def test_decides_answer_when_remaining_steps_low(self):
        """验证剩余步数少时强制 ANSWER。"""
        policy = RuleBasedPolicy()
        from src.schema.document import CandidateChunk, ReadSummary
        obs = create_test_observation(
            n_candidates=2,
            n_read=2,
            n_cited=1,
            remaining_steps=1,
        )
        obs.candidate_chunks = [
            CandidateChunk(chunk_id="c1", doc_id="d1", score=1.0, rank=1, query="test"),
        ]
        obs.read_summaries = [
            ReadSummary(chunk_id="c1", summary="test", key_claims=["claim1"], evidence_strength=0.8),
        ]

        action = policy.decide(obs)

        self.assertEqual(action.tool, "ANSWER")

    def test_decides_answer_when_no_progress(self):
        """验证无进展且有 citation 时决定 ANSWER。"""
        policy = RuleBasedPolicy()
        obs = create_test_observation(
            n_candidates=1,
            n_read=1,
            n_cited=2,
            remaining_steps=5,
            no_progress=2,
        )

        action = policy.decide(obs)

        self.assertEqual(action.tool, "ANSWER")


class TestRuleBasedPolicyEdgeCases(unittest.TestCase):
    """测试 edge cases。"""

    def test_fallback_to_answer(self):
        """验证 fallback 到 ANSWER。"""
        policy = RuleBasedPolicy()
        obs = create_test_observation(
            n_candidates=0,
            n_read=0,
            n_cited=0,
            remaining_steps=5,
            no_progress=0,
        )

        action = policy.decide(obs)

        self.assertEqual(action.tool, "SEARCH")


if __name__ == "__main__":
    unittest.main()
