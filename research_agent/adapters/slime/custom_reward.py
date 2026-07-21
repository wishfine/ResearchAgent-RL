from __future__ import annotations
from typing import Any
import os
import re
import string

from ...core.schema.parser import ActionParser

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
        
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(final_answer: str, ground_truth_answer: str, cited_ids: list[str], gt_citations: list[str]) -> dict:
    ans_clean = normalize_answer(final_answer) if final_answer else ""
    gt_clean = normalize_answer(ground_truth_answer) if ground_truth_answer else ""
    
    exact_match = 1.0 if ans_clean == gt_clean else 0.0
    contains = 1.0 if gt_clean in ans_clean else 0.0
    token_f1 = compute_token_f1(final_answer, ground_truth_answer)

    cited_set = set(cited_ids)
    gt_set = set(gt_citations)
    
    intersection = cited_set & gt_set
    precision = len(intersection) / len(cited_set) if cited_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    citation_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "contains": contains,
        "token_f1": token_f1,
        "citation_f1": citation_f1
    }


async def custom_rm(args: Any, sample: Any) -> float:
    """
    Asynchronous custom reward function for Slime.
    - sample: Slime sample containing the rollout prompt/response and metadata.
    """
    # 1. Retrieve task metadata (supports both object attributes and dict get)
    def get_val(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    metadata = get_val(sample, "metadata", {}) or {}

    def float_setting(attribute: str, environment: str, default: float) -> float:
        """Prefer an explicit framework argument, then an opt-in env setting."""
        value = get_val(args, attribute, None)
        if value is None:
            value = os.environ.get(environment, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    # Dataset adapters keep task-level verifier fields in ``metadata``.  Keep
    # the top-level lookup for backwards compatibility with older Slime data.
    ground_truth_answer = get_val(sample, "ground_truth_answer", "") or metadata.get("ground_truth_answer", "")
    ground_truth_citations = get_val(sample, "ground_truth_citations", []) or metadata.get(
        "ground_truth_citations", []
    )

    # 2. Extract final answer and citations
    final_answer = metadata.get("final_answer", None)
    cited_ids = metadata.get("cited_chunk_ids", None)
    steps_count = metadata.get("steps_count", 0)
    invalid_action_count = metadata.get("invalid_action_count", 0)
    valid_action_count = metadata.get("valid_action_count", max(0, steps_count - invalid_action_count))

    # Fallback: if metadata is empty, parse raw response text
    if final_answer is None or cited_ids is None:
        response_text = getattr(sample, "response", "") or sample.get("response", "")
        # Find all ANSWER action blocks in the response
        actions = []
        pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL)
        for block in pattern.findall(response_text):
            action = ActionParser.parse(f"<action>{block}</action>")
            if action.tool == "ANSWER":
                actions.append(action)
        
        if actions:
            # Get the last submitted answer
            last_ans = actions[-1]
            final_answer = last_ans.params.get("answer_text", "")
            cited_ids = last_ans.params.get("cited_chunk_ids", [])
        else:
            final_answer = ""
            cited_ids = []

    # 3. Compute accuracy and citation metrics
    metrics = compute_metrics(final_answer, ground_truth_answer, cited_ids, ground_truth_citations)

    # 4. Score answer quality and action protocol separately.  The latter is
    # essential for GRPO: without it an invalid trajectory can be nearly tied
    # with a grounded one when the group contains no correct answer.
    contains = metrics.get("contains", 0.0)
    token_f1 = metrics.get("token_f1", 0.0)
    citation_f1 = metrics.get("citation_f1", 0.0)
    invalid_penalty = float_setting("invalid_penalty", "RESEARCH_AGENT_INVALID_ACTION_PENALTY", 0.05)
    step_penalty = float_setting("step_penalty", "RESEARCH_AGENT_STEP_PENALTY", 0.01)
    format_reward = float_setting("format_reward", "RESEARCH_AGENT_FORMAT_REWARD", 0.0)
    format_valid_rate = valid_action_count / steps_count if steps_count else 0.0

    score = (1.0 * contains) + (0.5 * token_f1) + (0.5 * citation_f1)
    score += format_reward * format_valid_rate
    score -= invalid_penalty * invalid_action_count
    score -= step_penalty * steps_count

    # Clip to reasonable range
    return max(-2.0, min(2.0, score))
