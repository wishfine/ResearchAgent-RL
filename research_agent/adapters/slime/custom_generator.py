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
from ...core.tools.cite import CiteTool

# Module-level cache only for the CorpusStore to prevent reloading database in worker processes.
_cached_corpus: CorpusStore | None = None

def get_corpus(args: Any) -> CorpusStore:
    global _cached_corpus
    if _cached_corpus is None:
        corpus_dir = getattr(args, "corpus_dir", "data/searchqa_debug/corpus")
        _cached_corpus = CorpusStore(corpus_dir)
        _cached_corpus.load()
    return _cached_corpus


async def custom_generate(args: Any, sample: Any, sampling_params: dict, evaluation: bool = False) -> Any:
    """
    Asynchronous custom generation function for Slime.
    Modifies the passed-in sample object in-place and returns it.
    """
    # 1. Retrieve cached CorpusStore and create a fresh local ResearchEnv to prevent concurrency crosstalk
    corpus = get_corpus(args)
    max_steps = getattr(args, "max_steps", 10)
    env = ResearchEnv(corpus=corpus, max_steps=max_steps)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(CiteTool())
    env.register_tool(AnswerTool())

    # Helper function for safe retrieval from either objects or dictionaries
    def get_val(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # 2. Extract task information, prioritizing sample.metadata
    sample_metadata = get_val(sample, "metadata", {}) or {}
    
    task_id = get_val(sample_metadata, "task_id") or get_val(sample, "task_id", "unknown_task")
    user_query = get_val(sample_metadata, "user_query") or get_val(sample, "user_query", "")
    ground_truth_answer = get_val(sample_metadata, "ground_truth_answer") or get_val(sample, "ground_truth_answer", "")
    ground_truth_citations = get_val(sample_metadata, "ground_truth_citations") or get_val(sample, "ground_truth_citations", [])
    reference_docs = get_val(sample_metadata, "reference_docs") or get_val(sample, "reference_docs", [])

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
    actor_url = getattr(args, "actor_model_url", "http://localhost:8000/v1")
    is_mock = (actor_url == "mock")
    
    if not is_mock:
        client = LLMClient(
            api_url=actor_url,
            api_key=getattr(args, "api_key", None),
            model=getattr(args, "model", "Qwen3.5-9B"),
        )

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
        
        if is_mock:
            # Deterministic policy output (SEARCH -> READ -> CITE -> ANSWER)
            if step_count == 0:
                raw_response = '<reasoning>search</reasoning><action>{"tool": "SEARCH", "params": {"query": "Canada maple leaf", "topk": 1}}</action>'
            elif step_count == 1:
                raw_response = '<reasoning>read</reasoning><action>{"tool": "READ", "params": {"chunk_ids": ["maple_leaf_flag"]}}</action>'
            elif step_count == 2:
                raw_response = '<reasoning>cite</reasoning><action>{"tool": "CITE", "params": {"chunk_ids": ["maple_leaf_flag"], "claims": ["Canada has a red maple leaf flag."]}}</action>'
            else:
                raw_response = '<reasoning>answer</reasoning><action>{"tool": "ANSWER", "params": {"answer_text": "The answer is Canada.", "cited_chunk_ids": ["maple_leaf_flag"]}}</action>'
        else:
            # Query model using executor
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
    tokens, loss_mask = build_loss_mask(collector.segments, tokenizer)

    # The response part is everything after the initial prompt
    full_text = collector.get_full_text()
    response_text = full_text[len(prompt_text):]
    
    # Measure the response length in tokens as the sum of trainable loss mask tokens
    response_length = sum(loss_mask)

    # Keep original metadata keys if any, and append rollout details
    updated_metadata = dict(sample_metadata) if isinstance(sample_metadata, dict) else {}
    updated_metadata.update({
        "done_reason": episode_result.done_reason,
        "final_answer": episode_result.final_answer,
        "cited_chunk_ids": episode_result.cited_chunk_ids,
        "steps_count": step_count,
        "invalid_action_count": env._state.invalid_action_count,
        "valid_action_count": sum(step.is_valid for step in env._state.trajectory),
        "task_id": task_id,
        "ground_truth_answer": ground_truth_answer,
        "ground_truth_citations": ground_truth_citations
    })

    # 7. Modify the sample in-place
    if isinstance(sample, dict):
        sample["prompt"] = prompt_text
        sample["response"] = response_text
        sample["tokens"] = tokens
        sample["response_length"] = response_length
        sample["loss_mask"] = loss_mask
        sample["metadata"] = updated_metadata
    else:
        sample.prompt = prompt_text
        sample.response = response_text
        sample.tokens = tokens
        sample.response_length = response_length
        sample.loss_mask = loss_mask
        sample.metadata = updated_metadata

    return sample
