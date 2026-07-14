from __future__ import annotations
from typing import Any
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

    # 4. Calculate total reward score using user recommended formula:
    # reward = 1.0 * contains + 0.5 * token_f1 + 0.5 * citation_f1 - 0.05 * invalid_actions - 0.01 * steps
    contains = metrics.get("contains", 0.0)
    token_f1 = metrics.get("token_f1", 0.0)
    citation_f1 = metrics.get("citation_f1", 0.0)

    score = (1.0 * contains) + (0.5 * token_f1) + (0.5 * citation_f1)
    score -= (0.05 * invalid_action_count)
    score -= (0.01 * steps_count)

    # Clip to reasonable range
    return max(-2.0, min(2.0, score))
