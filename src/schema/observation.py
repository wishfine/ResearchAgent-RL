"""
src/schema/observation.py
职责: 定义 Observation——Environment 暴露给 Agent 的状态视图
设计: 结构化 17 字段，不泄露 corpus；current_subgoal/evidence_gap 驱动决策；remaining_steps/invalid_action_count/repeated_action_count/no_progress_count 用于终止判断
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .document import CandidateChunk, ReadSummary


@dataclass
class Observation:
    """
    Agent 看到的完整状态。
    """
    task_id: str
    task_type: str
    user_query: str
    current_subgoal: str = ""
    current_evidence_gap: str = ""
    candidate_chunks: List[CandidateChunk] = field(default_factory=list)
    read_summaries: List[ReadSummary] = field(default_factory=list)
    cited_chunks: List[str] = field(default_factory=list)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    failed_searches: List[str] = field(default_factory=list)
    trajectory: List[str] = field(default_factory=list)
    last_tool: Optional[str] = None
    last_tool_result_summary: str = ""
    remaining_steps: int = 15
    invalid_action_count: int = 0
    repeated_action_count: int = 0
    no_progress_count: int = 0
    final_answer: Optional[str] = None

    def get_decision_context(self) -> Dict[str, Any]:
        """
        供 rule-based baseline 使用的简化状态。
        """
        return {
            "remaining_steps": self.remaining_steps,
            "n_candidates": len(self.candidate_chunks),
            "n_read": len(self.read_summaries),
            "n_cited": len(self.cited_chunks),
            "no_progress": self.no_progress_count,
            "invalid": self.invalid_action_count,
            "last_tool": self.last_tool,
            "current_subgoal": self.current_subgoal,
            "current_evidence_gap": self.current_evidence_gap,
        }

    def has_new_candidate_since_last_step(self, prev_n_candidates: int) -> bool:
        return len(self.candidate_chunks) > prev_n_candidates

    def has_new_read_since_last_step(self, prev_n_read: int) -> bool:
        return len(self.read_summaries) > prev_n_read

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "user_query": self.user_query,
            "current_subgoal": self.current_subgoal,
            "current_evidence_gap": self.current_evidence_gap,
            "candidate_chunks": [c.to_dict() for c in self.candidate_chunks],
            "read_summaries": [s.to_dict() for s in self.read_summaries],
            "cited_chunks": self.cited_chunks,
            "search_history": self.search_history,
            "failed_searches": self.failed_searches,
            "trajectory": self.trajectory,
            "last_tool": self.last_tool,
            "last_tool_result_summary": self.last_tool_result_summary,
            "remaining_steps": self.remaining_steps,
            "invalid_action_count": self.invalid_action_count,
            "repeated_action_count": self.repeated_action_count,
            "no_progress_count": self.no_progress_count,
            "final_answer": self.final_answer,
        }
