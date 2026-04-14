"""
src/schema/task.py
职责: 定义任务样本和评测 rubric
设计: TaskType 枚举三种任务类型; Rubric 定义评测维度; TaskSample 包含完整任务信息 + ground truth 用于评测
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class TaskType(Enum):
    SURVEY_SYNTHESIS = "survey_synthesis"
    METHOD_COMPARISON = "method_comparison"
    EXPERIMENT_DESIGN = "experiment_design"


@dataclass
class Rubric:
    """
    评测规则：定义每个维度的评分标准和阈值。
    """
    citation_required: bool = True
    min_citations: int = 3
    max_citations: int = 20
    require_evidence_for_claims: bool = True
    penalize_hallucination: bool = True
    task_type_specific: dict = field(default_factory=dict)
    answer_weights: dict = field(default_factory=lambda: {
        "keyword_coverage": 0.4,
        "structural_completeness": 0.3,
        "no_hallucination": 0.3,
    })

    def to_dict(self) -> dict:
        return {
            "citation_required": self.citation_required,
            "min_citations": self.min_citations,
            "max_citations": self.max_citations,
            "require_evidence_for_claims": self.require_evidence_for_claims,
            "penalize_hallucination": self.penalize_hallucination,
            "task_type_specific": self.task_type_specific,
            "answer_weights": self.answer_weights,
        }


@dataclass
class TaskSample:
    """
    单个任务样本。
    """
    task_id: str
    task_type: TaskType
    user_query: str
    rubric: Rubric
    ground_truth_answer: Optional[str] = None
    ground_truth_citations: List[str] = field(default_factory=list)
    reference_docs: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    context: str = ""
    expected_subgoals: int = 3

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "user_query": self.user_query,
            "rubric": self.rubric.to_dict(),
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_citations": self.ground_truth_citations,
            "reference_docs": self.reference_docs,
            "difficulty": self.difficulty,
            "context": self.context,
            "expected_subgoals": self.expected_subgoals,
        }
