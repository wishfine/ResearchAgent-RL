"""
src/schema/result.py
职责: 定义 EpisodeResult, EvalResult, RewardSignals
设计: EpisodeResult 是轨迹 + 最终答案; EvalResult 是评测指标; RewardSignals 是 RL reward 接口
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EpisodeResult:
    """
    单个 episode 的完整记录。
    """
    task_id: str
    trajectory: List[Any] = field(default_factory=list)
    final_answer: str = ""
    cited_chunk_ids: List[str] = field(default_factory=list)
    done_reason: str = ""
    total_steps: int = 0
    n_valid_steps: int = 0
    n_invalid_steps: int = 0
    rewards: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "done_reason": self.done_reason,
            "total_steps": self.total_steps,
            "n_valid_steps": self.n_valid_steps,
            "n_invalid_steps": self.n_invalid_steps,
            "final_answer": self.final_answer,
            "cited_chunk_ids": self.cited_chunk_ids,
            "rewards": self.rewards,
            "trajectory": [s.to_dict() if hasattr(s, "to_dict") else str(s) for s in self.trajectory],
        }


@dataclass
class EvalResult:
    """
    单个 episode 的评测结果。
    """
    task_id: str = ""
    task_success: bool = False
    answer_quality: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    average_steps: float = 0.0
    repeated_query_rate: float = 0.0
    invalid_action_rate: float = 0.0
    search_success_rate: float = 1.0
    evidence_coverage: float = 0.0
    tool_diversity: float = 0.0
    answer_conciseness: float = 0.0
    episode_result: EpisodeResult = None
    rubric_scores: Dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_success": self.task_success,
            "answer_quality": round(self.answer_quality, 3),
            "citation_precision": round(self.citation_precision, 3),
            "citation_recall": round(self.citation_recall, 3),
            "citation_f1": round(self.citation_f1, 3),
            "total_steps": self.episode_result.total_steps,
            "tool_diversity": round(self.tool_diversity, 3),
            "repeated_query_rate": round(self.repeated_query_rate, 3),
            "search_success_rate": round(self.search_success_rate, 3),
        }

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_success": self.task_success,
            "answer_quality": self.answer_quality,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_f1": self.citation_f1,
            "average_steps": self.average_steps,
            "repeated_query_rate": self.repeated_query_rate,
            "invalid_action_rate": self.invalid_action_rate,
            "search_success_rate": self.search_success_rate,
            "evidence_coverage": self.evidence_coverage,
            "tool_diversity": self.tool_diversity,
            "answer_conciseness": self.answer_conciseness,
            "rubric_scores": self.rubric_scores,
            "episode_result": self.episode_result.to_dict(),
        }


@dataclass
class RewardSignals:
    """
    RL reward 信号分解。
    """
    final_answer_reward: float = 0.0
    citation_reward: float = 0.0
    evidence_gain_reward: float = 0.0
    repeated_action_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    efficiency_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "final_answer_reward": self.final_answer_reward,
            "citation_reward": self.citation_reward,
            "evidence_gain_reward": self.evidence_gain_reward,
            "repeated_action_penalty": self.repeated_action_penalty,
            "invalid_action_penalty": self.invalid_action_penalty,
            "efficiency_bonus": self.efficiency_bonus,
            "total": self.total,
        }
