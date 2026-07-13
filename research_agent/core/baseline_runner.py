from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable
import re

from research_agent.core.env.rollout_collector import ConversationCollector


def _token_f1(prediction: str, reference: str) -> float:
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", (text or "").lower())

    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _contains_reference(prediction: str, reference: str) -> float:
    predicted = " ".join(re.findall(r"\w+", (prediction or "").lower()))
    expected = " ".join(re.findall(r"\w+", (reference or "").lower()))
    return float(bool(expected) and expected in predicted)


def evaluate_episode(
    episode,
    task,
    search_history: list[dict],
    *,
    parsed_action_count: int | None = None,
    action_attempt_count: int | None = None,
) -> dict:
    cited = set(episode.cited_chunk_ids)
    expected_citations = set(task.ground_truth_citations)
    overlap = cited & expected_citations
    citation_precision = len(overlap) / len(cited) if cited else 0.0
    citation_recall = len(overlap) / len(expected_citations) if expected_citations else 0.0
    citation_f1 = (
        2 * citation_precision * citation_recall / (citation_precision + citation_recall)
        if citation_precision + citation_recall
        else 0.0
    )
    answer_token_f1 = _token_f1(episode.final_answer, task.ground_truth_answer or "")
    answer_contains = _contains_reference(episode.final_answer, task.ground_truth_answer or "")
    # Toy QA references are often short entities.  A complete generated answer
    # should not be scored below threshold merely because it contains context.
    answer_quality = max(answer_token_f1, answer_contains)
    search_queries = [item["query"] for item in search_history if item.get("query")]
    repeated_query_rate = (
        (len(search_queries) - len(set(search_queries))) / len(search_queries)
        if search_queries
        else 0.0
    )
    invalid_action_rate = episode.n_invalid_steps / episode.total_steps if episode.total_steps else 0.0
    if action_attempt_count is None:
        action_attempt_count = episode.total_steps
    if parsed_action_count is None:
        # Backward-compatible fallback for records generated before parser and
        # environment validity were tracked separately.
        parsed_action_count = action_attempt_count - episode.n_invalid_steps
    action_parse_success_rate = (
        parsed_action_count / action_attempt_count if action_attempt_count else 0.0
    )
    task_success = answer_quality >= 0.5 and citation_recall >= 0.3
    reward = (
        answer_quality
        + 0.5 * citation_f1
        - 0.05 * episode.n_invalid_steps
        - 0.01 * episode.total_steps
    )
    return {
        "action_parse_success_rate": action_parse_success_rate,
        "invalid_action_rate": invalid_action_rate,
        "task_success": task_success,
        "answer_quality": answer_quality,
        "answer_contains": answer_contains,
        "answer_token_f1": answer_token_f1,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": citation_f1,
        "average_steps": float(episode.total_steps),
        "repeated_query_rate": repeated_query_rate,
        "reward": reward,
    }


def run_episode(
    env,
    task,
    actor,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float | None = None,
) -> dict:
    """Run one LLM-controlled environment episode and return a JSONL-ready record."""
    observation = env.reset(task)
    collector = ConversationCollector()
    collector.add_user_message(task.user_query)
    steps: list[dict] = []
    total_latency = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    parsed_action_count = 0
    action_attempt_count = 0
    termination_reason: str | None = None

    while termination_reason is None:
        prompt = collector.get_prompt_for_generation()
        observation_before = observation.to_dict()
        try:
            turn = actor.decide(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_tokens=["<|im_end|>"],
            )
        except Exception as exc:
            steps.append(
                {
                    "observation": observation_before,
                    "action": None,
                    "tool_result": None,
                    "reasoning": "",
                    "raw_content": "",
                    "latency_sec": 0.0,
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "error": f"actor_error: {exc}",
                }
            )
            termination_reason = "actor_error"
            break

        total_latency += turn.latency_sec
        prompt_tokens += turn.prompt_tokens
        completion_tokens += turn.completion_tokens
        action_attempt_count += 1
        parsed_action_count += int(turn.action.tool != "INVALID")
        observation, done, reason = env.step(turn.action)
        trajectory_step = env._state.trajectory[-1]
        steps.append(
            {
                "observation": observation_before,
                "action": turn.action.to_dict(),
                "tool_result": trajectory_step.tool_result.to_dict()
                if trajectory_step.tool_result is not None
                else None,
                "reasoning": turn.reasoning,
                "raw_content": turn.raw_content,
                "latency_sec": turn.latency_sec,
                "token_usage": {
                    "prompt_tokens": turn.prompt_tokens,
                    "completion_tokens": turn.completion_tokens,
                },
                "error": trajectory_step.error_message,
            }
        )
        collector.add_assistant_response(turn.raw_content)

        if done:
            termination_reason = reason
        else:
            result = trajectory_step.tool_result
            collector.add_observation(
                turn.action.tool,
                bool(trajectory_step.is_valid and result is not None and result.success),
                result.data if result is not None and result.success else {},
                trajectory_step.error_message if not trajectory_step.is_valid else (result.error if result else ""),
            )

    episode = env.finalize_episode()
    if termination_reason == "actor_error":
        episode.done_reason = termination_reason
    metrics = evaluate_episode(
        episode,
        task,
        env._state.search_history,
        parsed_action_count=parsed_action_count,
        action_attempt_count=action_attempt_count,
    )
    messages = [{"role": segment.role, "content": segment.text} for segment in collector.segments]
    return {
        "task_id": task.task_id,
        "messages": messages,
        "observations": [step["observation"] for step in steps],
        "actions": [step["action"] for step in steps],
        "tool_results": [step["tool_result"] for step in steps],
        "reasoning": [step["reasoning"] for step in steps],
        "steps": steps,
        "final_answer": episode.final_answer,
        "reward": metrics["reward"],
        "eval_metrics": metrics,
        "termination_reason": episode.done_reason,
        "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "latency_sec": total_latency,
    }


def aggregate_records(records: Iterable[dict]) -> dict:
    records = list(records)
    if not records:
        return {"episodes": 0}

    metric_names = [
        "action_parse_success_rate",
        "invalid_action_rate",
        "answer_quality",
        "answer_contains",
        "answer_token_f1",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "average_steps",
        "repeated_query_rate",
    ]
    summary = {"episodes": len(records)}
    for name in metric_names:
        summary[name] = sum(record["eval_metrics"][name] for record in records) / len(records)
    summary["task_success"] = sum(bool(record["eval_metrics"]["task_success"]) for record in records) / len(records)
    summary["average_prompt_tokens"] = sum(
        record["token_usage"]["prompt_tokens"] for record in records
    ) / len(records)
    summary["average_completion_tokens"] = sum(
        record["token_usage"]["completion_tokens"] for record in records
    ) / len(records)
    summary["average_episode_latency_sec"] = sum(record["latency_sec"] for record in records) / len(records)
    return summary
