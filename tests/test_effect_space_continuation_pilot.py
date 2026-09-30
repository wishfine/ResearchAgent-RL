from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.baselines.llm_actor import ActorTurn
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.schema.action import Action
from scripts.effect_space_continuation_pilot import (
    ForcedFirstSearchActor,
    analyze_continuations,
    canonical_action_text,
    prepare_manifest,
    run_variant,
)


def raw_action(action: Action) -> str:
    return "<action>" + json.dumps({
        "tool": action.tool,
        "intent": action.intent,
        "params": action.params,
    }) + "</action>"


def sample(task_id: str, index: int, query: str, topk: int = 2) -> dict:
    return {
        "task_id": task_id, "sample_index": index, "valid_search": True,
        "query": query, "topk": topk,
        "raw_content": raw_action(Action.search(query, topk=topk)),
    }


def test_prepare_selects_label_blind_pairs_with_fixed_topk(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    task_path = tasks / "task_1.json"
    task = {
        "task_id": "task_1", "retrieval_scope": "split_corpus",
        "reference_docs": [], "ground_truth_answer": "PRIVATE_LABEL",
        "ground_truth_citations": ["SECRET_CHUNK"],
    }
    task_path.write_text(json.dumps(task), encoding="utf-8")

    class FakeCorpus:
        def search(self, query, topk, doc_ids):
            assert doc_ids is None
            chunk_id = "shared" if query in {"first", "second"} else "other"
            return [SimpleNamespace(chunk_id=chunk_id, score=1.0)]

    rows = [
        sample("task_1", 0, "first"),
        sample("task_1", 1, "second"),
        sample("task_1", 2, "third"),
        sample("task_1", 3, "first", topk=3),
    ]
    before = prepare_manifest(rows, tasks, FakeCorpus(), max_tasks=1,
                              max_pairs_per_task=1, seed=42)
    task["ground_truth_answer"] = "DIFFERENT_SECRET"
    task["ground_truth_citations"] = ["OTHER_SECRET_CHUNK"]
    task_path.write_text(json.dumps(task), encoding="utf-8")
    after = prepare_manifest(rows, tasks, FakeCorpus(), max_tasks=1,
                             max_pairs_per_task=1, seed=42)
    assert before == after
    assert before["selected_tasks"] == 1
    pairs = before["tasks"][0]["pairs"]
    assert any(pair["kind"] == "within" and
               {pair["left"], pair["right"]} == {0, 1} for pair in pairs)
    assert any(pair["kind"] == "between" for pair in pairs)
    assert "PRIVATE_LABEL" not in json.dumps(before)


def test_prepare_rejects_raw_action_mismatch(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "task_1.json").write_text(
        json.dumps({"task_id": "task_1", "retrieval_scope": "split_corpus"}),
        encoding="utf-8",
    )

    class FakeCorpus:
        def search(self, query, topk, doc_ids):
            return [SimpleNamespace(chunk_id="shared", score=1.0)]

    wrong = sample("task_1", 0, "first")
    wrong["raw_content"] = raw_action(Action.search("different", topk=2))
    with pytest.raises(ValueError, match="disagrees"):
        prepare_manifest([wrong, sample("task_1", 1, "second")],
                         tasks, FakeCorpus(), max_tasks=1,
                         max_pairs_per_task=1, seed=42)


class SequenceActor:
    def __init__(self, actions: list[Action]):
        self.actions = iter(actions)

    def decide(self, prompt, **kwargs) -> ActorTurn:
        action = next(self.actions)
        return ActorTurn(
            raw_content=raw_action(action), action=action, reasoning="",
            latency_sec=0.0, prompt_tokens=10, completion_tokens=5,
        )


def test_forced_query_executes_original_while_canonicalizing_history(tmp_path: Path) -> None:
    corpus = CorpusStore(
        str(Path(__file__).resolve().parents[1] / "data/searchqa_debug/corpus")
    )
    corpus.load()
    task_path = tmp_path / "task_1.json"
    task_path.write_text(json.dumps({
        "task_id": "task_1",
        "user_query": "Which country's flag features a red maple leaf?",
        "ground_truth_answer": "Canada",
        "ground_truth_citations": ["maple_leaf_flag"],
        "retrieval_scope": "split_corpus",
    }), encoding="utf-8")
    variant = sample("task_1", 0, "Canada maple leaf", topk=1)
    variant["effect_chunk_ids"] = [
        item.chunk_id for item in corpus.search("Canada maple leaf", topk=1)
    ]
    assert variant["effect_chunk_ids"] == ["maple_leaf_flag"]
    actor = SequenceActor([
        Action.read(["maple_leaf_flag"]),
        Action.cite(["maple_leaf_flag"], ["Canada has a maple leaf flag"]),
        Action.answer("Canada", ["maple_leaf_flag"]),
    ])
    try:
        record = run_variant(
            task_path, corpus, actor, variant,
            history_mode="canonical", canonical_query="maple flag Canada",
            max_steps=4, max_tokens=64, temperature=0.7, top_p=0.95,
        )
    finally:
        corpus.close()
    assert record["eval_metrics"]["task_success"]
    assert record["steps"][0]["action"]["params"]["query"] == "Canada maple leaf"
    assert record["messages"][2]["content"] == canonical_action_text(
        "maple flag Canada", 1
    )
    assert record["token_usage"]["prompt_tokens"] == 30  # Forced action not regenerated.


def test_analysis_separates_original_and_canonical_residuals() -> None:
    manifest = {
        "tasks": [{
            "task_id": "task_1",
            "pairs": [
                {"kind": "within", "left": 0, "right": 1},
                {"kind": "between", "left": 0, "right": 2},
            ],
        }],
    }

    def record(reward: float) -> dict:
        return {
            "reward": reward,
            "eval_metrics": {
                "answer_quality": reward,
                "citation_recall": reward,
                "task_success": reward >= 0.5,
            },
        }

    runs = []
    for mode, values in (
        ("original", {0: 0.2, 1: 0.4, 2: 1.0}),
        ("canonical", {0: 0.3, 1: 0.3, 2: 1.0}),
    ):
        for index, reward in values.items():
            runs.append({
                "sample_index": index, "repeat": 0,
                "history_mode": mode, "record": record(reward),
            })
    summary = analyze_continuations(
        manifest, {"task_1": {"task_id": "task_1", "runs": runs}}
    )
    assert summary["complete"]
    assert summary["mean_within_original_reward_gap"] == pytest.approx(0.2)
    assert summary["mean_within_canonical_reward_gap"] == pytest.approx(0.0)
    assert summary["mean_between_original_reward_gap"] == pytest.approx(0.8)
    assert summary["mean_paired_between_minus_within"] == pytest.approx(0.6)
    assert summary["mean_within_original_minus_canonical"] == pytest.approx(0.2)
    assert summary["paired_between_minus_within_bootstrap_ci95"] is None


def test_analysis_reports_split_half_sampling_noise_and_score_shift() -> None:
    manifest = {
        "tasks": [{
            "task_id": "task_1",
            "variants": [
                {"sample_index": 0, "candidate_scores": [1.0, 0.5]},
                {"sample_index": 1, "candidate_scores": [0.8, 0.4]},
            ],
            "pairs": [{"kind": "within", "left": 0, "right": 1}],
        }],
    }

    def record(reward: float) -> dict:
        return {
            "reward": reward,
            "eval_metrics": {
                "answer_quality": reward,
                "citation_recall": reward,
                "task_success": reward >= 0.5,
            },
        }

    runs = [
        {"sample_index": sample_index, "repeat": repeat,
         "history_mode": mode, "record": record(reward)}
        for mode in ("original", "canonical")
        for sample_index in (0, 1)
        for repeat, reward in enumerate((0.0, 1.0, 0.0, 1.0))
    ]
    summary = analyze_continuations(
        manifest, {"task_1": {"task_id": "task_1", "runs": runs}}
    )
    assert summary["mean_within_original_reward_gap"] == pytest.approx(0.0)
    assert summary["mean_original_split_half_reward_noise_floor"] == pytest.approx(1.0)
    assert summary["mean_canonical_split_half_reward_noise_floor"] == pytest.approx(1.0)
    assert summary["pairs"][0]["max_retrieval_score_shift"] == pytest.approx(0.2)


def test_forced_actor_rejects_manifest_action_mismatch() -> None:
    variant = sample("task_1", 0, "query a")
    variant["query"] = "query b"
    forced = ForcedFirstSearchActor(SequenceActor([]), variant, variant["raw_content"])
    with pytest.raises(ValueError, match="disagrees"):
        forced.decide("prompt")
