"""
src/data/loader.py
职责: Task 和 Corpus 的加载器
设计: 从 YAML/JSON 加载 TaskSample；加载 CorpusStore
"""
from __future__ import annotations
import os
import json
import yaml
from typing import List, Optional
from ..schema.task import TaskSample, Rubric, TaskType
from ..environment.corpus import CorpusStore


class TaskLoader:
    """加载任务数据（YAML/JSON）。"""

    def load_file(self, filepath: str) -> TaskSample:
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return self._parse(data)

    def load_dir(self, dirpath: str) -> List[TaskSample]:
        samples = []
        for filename in os.listdir(dirpath):
            if filename.startswith("."):
                continue
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                try:
                    samples.append(self.load_file(filepath))
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")
        return samples

    def _parse(self, data: dict) -> TaskSample:
        rubric_data = data.get("rubric", {})
        rubric = Rubric(
            citation_required=rubric_data.get("citation_required", True),
            min_citations=rubric_data.get("min_citations", 3),
            max_citations=rubric_data.get("max_citations", 20),
            require_evidence_for_claims=rubric_data.get("require_evidence_for_claims", True),
            penalize_hallucination=rubric_data.get("penalize_hallucination", True),
            task_type_specific=rubric_data.get("task_type_specific", {}),
            answer_weights=rubric_data.get("answer_weights", {
                "keyword_coverage": 0.4,
                "structural_completeness": 0.3,
                "no_hallucination": 0.3,
            }),
        )
        return TaskSample(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            user_query=data["user_query"],
            rubric=rubric,
            ground_truth_answer=data.get("ground_truth_answer"),
            ground_truth_citations=data.get("ground_truth_citations", []),
            reference_docs=data.get("reference_docs", []),
            difficulty=data.get("difficulty", "medium"),
            context=data.get("context", ""),
            expected_subgoals=data.get("expected_subgoals", 3),
        )


class CorpusLoader:
    """加载 CorpusStore。"""

    def load(self, corpus_dir: str = "data/corpus") -> CorpusStore:
        corpus = CorpusStore(corpus_dir)
        corpus.load()
        return corpus
