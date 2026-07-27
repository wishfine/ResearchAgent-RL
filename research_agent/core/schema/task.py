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
    citation_required: bool = True
    min_citations: int = 2
    max_citations: int = 10
    require_evidence_for_claims: bool = True
    penalize_hallucination: bool = True
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
            "answer_weights": self.answer_weights,
        }

@dataclass
class TaskSample:
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
            "task_type": self.task_type.value if hasattr(self.task_type, "value") else str(self.task_type),
            "user_query": self.user_query,
            "rubric": self.rubric.to_dict() if hasattr(self.rubric, "to_dict") else self.rubric,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_citations": self.ground_truth_citations,
            "reference_docs": self.reference_docs,
            "difficulty": self.difficulty,
            "context": self.context,
            "expected_subgoals": self.expected_subgoals,
        }
