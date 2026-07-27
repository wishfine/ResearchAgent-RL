"""Multi-turn ResearchEnv rollout hooks for Vime.

The hook deliberately talks to Vime's in-cluster vLLM router through the
framework's own ``generate`` helper.  It therefore records rollout token IDs
and log-probabilities, unlike the legacy Slime adapter which calls an external
OpenAI-compatible endpoint.

Configuration is intentionally environment based because stock Vime's CLI is
owned by Megatron and does not expose project-specific arguments::

    RESEARCH_AGENT_CORPUS_DIR=/data/.../hotpotqa_7k3k/corpus/train
    RESEARCH_AGENT_MAX_STEPS=6
"""

from __future__ import annotations

import copy
import os
from typing import Any

from research_agent.adapters.slime.custom_reward import custom_rm
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.env import ResearchEnv
from research_agent.core.env.rollout_collector import ConversationCollector
from research_agent.core.schema.parser import ActionParser
from research_agent.core.schema.task import Rubric, TaskSample, TaskType
from research_agent.core.tools.answer import AnswerTool
from research_agent.core.tools.cite import CiteTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.rerank import RerankTool
from research_agent.core.tools.search import SearchTool

_cached_corpus: CorpusStore | None = None
_cached_corpus_dir: str | None = None


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _corpus_dir(args: Any) -> str:
    return os.environ.get(
        "RESEARCH_AGENT_CORPUS_DIR", getattr(args, "corpus_dir", "data/searchqa_debug/corpus")
    )


def _get_corpus(args: Any) -> CorpusStore:
    global _cached_corpus, _cached_corpus_dir
    corpus_dir = _corpus_dir(args)
    if _cached_corpus is None or _cached_corpus_dir != corpus_dir:
        corpus = CorpusStore(corpus_dir)
        corpus.load()
        _cached_corpus = corpus
        _cached_corpus_dir = corpus_dir
    return _cached_corpus


def _make_env(args: Any) -> ResearchEnv:
    max_steps = int(os.environ.get("RESEARCH_AGENT_MAX_STEPS", "6"))
    env = ResearchEnv(corpus=_get_corpus(args), max_steps=max_steps)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(RerankTool())
    env.register_tool(CiteTool())
    env.register_tool(AnswerTool())
    return env


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode without adding a BOS token at every agent-turn boundary."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _task_from_sample(sample: Any) -> tuple[TaskSample, dict[str, Any]]:
    metadata = dict(_get_value(sample, "metadata", {}) or {})
    task_id = metadata.get("task_id") or _get_value(sample, "task_id", "unknown_task")
    query = metadata.get("user_query") or _get_value(sample, "user_query", "")
    answer = metadata.get("ground_truth_answer") or _get_value(sample, "ground_truth_answer", "")
    citations = metadata.get("ground_truth_citations") or _get_value(sample, "ground_truth_citations", [])
    reference_docs = metadata.get("reference_docs") or _get_value(sample, "reference_docs", [])
    if not query:
        raise ValueError(f"{task_id} has no user_query in sample metadata")
    return (
        TaskSample(
            task_id=task_id,
            task_type=TaskType.SURVEY_SYNTHESIS,
            user_query=query,
            rubric=Rubric(),
            ground_truth_answer=answer,
            ground_truth_citations=citations,
            reference_docs=reference_docs,
        ),
        metadata,
    )


def _new_turn_sample(sample: Any, prompt: str, sample_cls: Any) -> Any:
    """Create a clean one-turn Vime sample while preserving routing metadata."""
    turn = copy.copy(sample)
    turn.prompt = prompt
    turn.tokens = []
    turn.response = ""
    turn.response_length = 0
    turn.loss_mask = None
    turn.rollout_log_probs = None
    turn.reward = None
    turn.status = sample_cls.Status.PENDING
    return turn


async def custom_generate(args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False) -> Any:
    """Execute SEARCH/READ/RERANK/CITE/ANSWER turns through Vime's router.

    Every model-generated assistant action remains trainable.  ChatML wrappers,
    tool observations, and the next assistant prefix are appended with a zero
    loss mask and zero rollout log-probability, yielding a Vime-compatible
    multi-turn trajectory.
    """
    # Import lazily so project unit tests can run without a Vime installation.
    from vime.rollout.vllm_rollout import GenerateState, generate as vime_generate
    from vime.utils.types import Sample

    task, metadata = _task_from_sample(sample)
    env = _make_env(args)
    env.reset(task)
    collector = ConversationCollector()
    collector.add_user_message(task.user_query)

    state = GenerateState(args)
    tokenizer = state.tokenizer
    initial_prompt = collector.get_prompt_for_generation()
    initial_tokens = _encode(tokenizer, initial_prompt)

    response_tokens: list[int] = []
    response_loss_mask: list[int] = []
    response_log_probs: list[float] = []
    response_text_parts: list[str] = []

    def append(ids: list[int], *, trainable: bool, text: str, log_probs: list[float] | None = None) -> None:
        if trainable:
            if log_probs is None or len(log_probs) != len(ids):
                raise ValueError("Vime trainable action tokens require aligned rollout log-probabilities")
            response_log_probs.extend(float(value) for value in log_probs)
        else:
            response_log_probs.extend([0.0] * len(ids))
        response_tokens.extend(ids)
        response_loss_mask.extend([1 if trainable else 0] * len(ids))
        response_text_parts.append(text)

    done = False
    saw_truncation = False
    generation_error: str | None = None
    while not done:
        prompt = collector.get_prompt_for_generation()
        try:
            turn = _new_turn_sample(sample, prompt, Sample)
            turn = await vime_generate(args, turn, sampling_params)
        except Exception as exc:  # Router failures should not become malformed training samples.
            generation_error = f"router_generation_error: {type(exc).__name__}: {exc}"
            break

        generated_ids = list(turn.tokens[-turn.response_length :]) if turn.response_length else []
        generated_log_probs = list(turn.rollout_log_probs or [])
        append(generated_ids, trainable=True, text=turn.response, log_probs=generated_log_probs)
        collector.add_assistant_response(turn.response)
        saw_truncation = saw_truncation or turn.status == Sample.Status.TRUNCATED

        action = ActionParser.parse(turn.response)
        _, done, _ = env.step(action)
        step = env._state.trajectory[-1]

        # Close the assistant turn in the supervised transcript.  The model did
        # not generate this structural token, so it must be masked out.
        append(_encode(tokenizer, "<|im_end|>\n"), trainable=False, text="<|im_end|>\n")
        if not done:
            result = step.tool_result
            collector.add_observation(
                action.tool,
                bool(step.is_valid and result is not None and result.success),
                result.data if result is not None and result.success else {},
                step.error_message if not step.is_valid else (result.error if result else ""),
            )
            observation_text = collector.segments[-1].to_chatml()
            append(_encode(tokenizer, observation_text), trainable=False, text=observation_text)
            assistant_prefix = "<|im_start|>assistant\n"
            append(_encode(tokenizer, assistant_prefix), trainable=False, text=assistant_prefix)

    episode = env.finalize_episode()
    updated_metadata = dict(metadata)
    updated_metadata.update(
        {
            "task_id": task.task_id,
            "ground_truth_answer": task.ground_truth_answer,
            "ground_truth_citations": task.ground_truth_citations,
            "done_reason": episode.done_reason,
            "final_answer": episode.final_answer,
            "cited_chunk_ids": episode.cited_chunk_ids,
            "steps_count": episode.total_steps,
            "invalid_action_count": episode.n_invalid_steps,
            "valid_action_count": episode.n_valid_steps,
            "format_valid_rate": (
                episode.n_valid_steps / episode.total_steps if episode.total_steps else 0.0
            ),
            "generation_error": generation_error,
        }
    )

    sample.prompt = initial_prompt
    sample.tokens = initial_tokens + response_tokens
    sample.response = "".join(response_text_parts)
    sample.response_length = len(response_tokens)
    sample.loss_mask = response_loss_mask
    sample.rollout_log_probs = response_log_probs
    sample.metadata = updated_metadata
    sample.reward = None
    sample.status = Sample.Status.ABORTED if generation_error else (
        Sample.Status.TRUNCATED if saw_truncation else Sample.Status.COMPLETED
    )
    return sample
