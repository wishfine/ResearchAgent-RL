from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.core.corpus.store import CorpusStore
from scripts.decision_value_pilot import (
    analyze,
    answer_distribution,
    chat_prompt,
    collect_task,
    parse_answer,
    parse_queries,
)


def test_query_parsing_keeps_non_oracle_question_and_deduplicates() -> None:
    result = parse_queries(
        '{"queries": ["Who founded Williams College?", "where is williams college", '
        '"  WHERE is Williams College  "]}',
        "Who founded Williams College?",
        3,
    )
    assert result == ["Who founded Williams College?", "where is williams college"]


def test_answer_parsing_does_not_silently_repair_bad_format() -> None:
    assert parse_answer('{"answer": "Mark Raymond"}') == "Mark Raymond"
    assert parse_answer("Mark Raymond") == ""


def test_sample_entropy_and_mode() -> None:
    mode, entropy = answer_distribution(["Paris", "Paris", "London", "London"])
    assert mode == "london"  # deterministic tie break, not sample order
    assert entropy == pytest.approx(0.6931471805599453)


def test_entropy_selection_can_increase_decision_error() -> None:
    task = {"ground_truth_answer": "Paris", "ground_truth_answer_aliases": []}
    prior = ["Paris", "London", "London", "Paris"]
    rows = [
        {"task_id": "q1", "task": task, "query": "misleading",
         "prior_samples": prior, "posterior_samples": ["London"] * 4,
         "retrieved_chunk_ids": ["c1"]},
        {"task_id": "q1", "task": task, "query": "useful",
         "prior_samples": prior, "posterior_samples": ["Paris"] * 3 + ["London"],
         "retrieved_chunk_ids": ["c2"]},
    ]
    summary = analyze(rows)
    assert summary["comparable_tasks"] == 1
    assert summary["entropy_ranking_regret_rate"] == 1.0
    assert summary["per_task"][0]["entropy_selected_query"] == "misleading"
    assert summary["per_task"][0]["quality_best_query_oracle"] == "useful"


def test_inconsistent_prior_rejected() -> None:
    task = {"ground_truth_answer": "Paris"}
    with pytest.raises(ValueError, match="Inconsistent prior"):
        analyze([
            {"task_id": "q1", "task": task, "query": "a",
             "prior_samples": ["Paris"], "posterior_samples": ["Paris"],
             "retrieved_chunk_ids": []},
            {"task_id": "q1", "task": task, "query": "b",
             "prior_samples": ["London"], "posterior_samples": ["London"],
             "retrieved_chunk_ids": []},
        ])


def test_unparseable_answer_is_excluded_from_comparison() -> None:
    task = {"ground_truth_answer": "Paris"}
    summary = analyze([
        {"task_id": "q1", "task": task, "query": "a",
         "prior_samples": ["Paris"], "posterior_samples": [""],
         "retrieved_chunk_ids": []},
        {"task_id": "q1", "task": task, "query": "b",
         "prior_samples": ["Paris"], "posterior_samples": ["Paris"],
         "retrieved_chunk_ids": []},
    ])
    assert summary["comparable_tasks"] == 0
    assert summary["answer_parse_success_rate"] == pytest.approx(2 / 3)
    assert summary["per_task"][0]["reason"] == "unparseable_answer_sample"


def test_prompt_builder_cannot_insert_ground_truth_implicitly() -> None:
    prompt = chat_prompt("Return an answer", "Question: A?")
    assert "Question: A?" in prompt
    assert "ground_truth" not in prompt


def test_collect_task_never_sends_ground_truth_to_model() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.prompts = []

        def generate_response(self, prompt, **kwargs):
            self.prompts.append(prompt)
            if "Produce distinct corpus search queries" in prompt:
                content = '{"queries": ["red maple leaf flag", "country flag maple"]}'
            else:
                content = '{"answer": "Canada"}'
            return SimpleNamespace(content=content, prompt_tokens=10, completion_tokens=5)

    corpus = CorpusStore(str(Path(__file__).resolve().parents[1] /
                             "data/searchqa_debug/corpus"))
    corpus.load()
    client = FakeClient()
    try:
        rows = collect_task(
            {"task_id": "secret_task", "user_query": "Which flag has a maple leaf?",
             "ground_truth_answer": "PRIVATE_LABEL", "reference_docs": ["toy_wiki_corpus"]},
            corpus, client, answer_count=2, max_queries=3, topk=2, temperature=0.7,
        )
    finally:
        corpus.close()
    assert len(rows) == 3
    assert all("PRIVATE_LABEL" not in prompt for prompt in client.prompts)
    assert any(row["retrieved_chunk_ids"] for row in rows)
