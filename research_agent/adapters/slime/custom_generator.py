from __future__ import annotations
import asyncio
from typing import Any, List, Dict
import os

from ...core.env.env import ResearchEnv
from ...core.corpus.store import CorpusStore
from ...core.schema.task import TaskSample, Rubric, TaskType
from ...core.schema.parser import ActionParser
from ...core.env.llm_client import LLMClient
from ...core.env.rollout_collector import ConversationCollector
from ...core.env.mask_builder import build_loss_mask, MockTokenizer
from ...core.tools.search import SearchTool
from ...core.tools.read import ReadTool
from ...core.tools.answer import AnswerTool

# Module-level cache to prevent reloading corpus on every worker generation call
_cached_env: ResearchEnv | None = None
_cached_corpus: CorpusStore | None = None

def get_env(args: Any) -> ResearchEnv:
    global _cached_env, _cached_corpus
    if _cached_env is None:
        corpus_dir = getattr(args, "corpus_dir", "data/searchqa_debug/corpus")
        _cached_corpus = CorpusStore(corpus_dir)
        _cached_corpus.load()
        
        max_steps = getattr(args, "max_steps", 10)
        _cached_env = ResearchEnv(corpus=_cached_corpus, max_steps=max_steps)
        _cached_env.register_tool(SearchTool())
        _cached_env.register_tool(ReadTool())
        _cached_env.register_tool(AnswerTool())
    return _cached_env


class SlimeSample:
    """A container for the generated episode rollout that is duck-type compatible with Slime's Sample."""
    def __init__(self, prompt: str, response: str, input_ids: List[int], loss_mask: List[int], task_id: str, metadata: Dict[str, Any] = None):
        self.prompt = prompt
        self.response = response
        self.input_ids = input_ids
        self.loss_mask = loss_mask
        self.task_id = task_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "input_ids": self.input_ids,
            "loss_mask": self.loss_mask,
            "task_id": self.task_id,
            "metadata": self.metadata
        }


async def custom_generate(args: Any, sample: Any, sampling_params: dict) -> SlimeSample:
    """
    Asynchronous custom generation function for Slime.
    - sample: must contain task metadata (task_id, user_query, and optional ground_truth/reference_docs)
    - sampling_params: generation configurations (temperature, max_tokens, tokenizer, etc.)
    """
    # 1. Retrieve cached environment
    env = get_env(args)

    # 2. Extract task information from sample (supports both object attributes and dict get)
    def get_val(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    task_id = get_val(sample, "task_id", "unknown_task")
    user_query = get_val(sample, "user_query", "")
    ground_truth_answer = get_val(sample, "ground_truth_answer", "")
    ground_truth_citations = get_val(sample, "ground_truth_citations", [])
    reference_docs = get_val(sample, "reference_docs", [])

    task = TaskSample(
        task_id=task_id,
        task_type=TaskType.SURVEY_SYNTHESIS,
        user_query=user_query,
        rubric=Rubric(),
        ground_truth_answer=ground_truth_answer,
        ground_truth_citations=ground_truth_citations,
        reference_docs=reference_docs
    )

    # 3. Setup LLM Client targeting the current Actor model URL
    # Slime typically exposes the local actor address via args.actor_model_url or env variables
    actor_url = getattr(args, "actor_model_url", "http://localhost:8000/v1")
    client = LLMClient(api_url=actor_url, api_key=getattr(args, "api_key", None))

    # 4. Initialize rollout and context collectors
    obs = env.reset(task)
    collector = ConversationCollector()
    collector.add_user_message(user_query)

    # Prompt is the initial system + user query text block
    prompt_text = collector.segments[0].to_chatml() + collector.segments[1].to_chatml()

    done = False
    step_count = 0
    max_tokens = sampling_params.get("max_tokens", 512)
    temperature = sampling_params.get("temperature", 0.0)

    # 5. Rollout loop
    while not done:
        prompt = collector.get_prompt_for_generation()
        
        # Query model (using run_in_executor to avoid blocking event loop on HTTP requests)
        loop = asyncio.get_event_loop()
        raw_response = await loop.run_in_executor(
            None,
            lambda: client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_tokens=["<|im_end|>", "</action>"]
            )
        )

        if not raw_response.endswith("</action>") and "<action>" in raw_response:
            raw_response += "</action>"

        action = ActionParser.parse(raw_response)
        
        # Step env
        obs, done, reason = env.step(action)
        step_count += 1

        # Record assistant turn
        collector.add_assistant_response(raw_response)

        # Record observation turn if not done
        if not done:
            last_step = env._state.trajectory[-1]
            success = last_step.is_valid and last_step.tool_result is not None and last_step.tool_result.success
            error = last_step.error_message if not last_step.is_valid else (last_step.tool_result.error if last_step.tool_result else "")
            data = last_step.tool_result.data if (last_step.tool_result and last_step.tool_result.success) else {}
            collector.add_observation(action.tool, success, data, error)

    # Finalize episode results
    episode_result = env.finalize_episode()

    # 6. Compute tokens and loss masks
    tokenizer = getattr(args, "tokenizer", None) or sampling_params.get("tokenizer", None) or MockTokenizer()
    input_ids, loss_mask = build_loss_mask(collector.segments, tokenizer)

    # The response part is everything after the initial prompt
    full_text = collector.get_full_text()
    response_text = full_text[len(prompt_text):]

    # Return the Slime compatible Sample
    return SlimeSample(
        prompt=prompt_text,
        response=response_text,
        input_ids=input_ids,
        loss_mask=loss_mask,
        task_id=task_id,
        metadata={
            "done_reason": episode_result.done_reason,
            "final_answer": episode_result.final_answer,
            "cited_chunk_ids": episode_result.cited_chunk_ids,
            "steps_count": step_count,
            "invalid_action_count": env._state.invalid_action_count
        }
    )
