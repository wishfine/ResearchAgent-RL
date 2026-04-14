"""
src/eval/metrics.py
职责: 计算评测指标
设计: citation-aware 是核心差异点；outcome + behavior + process 三类指标
修复:
  1. repeated_query_rate 计算重复 query（同一 query 多次 SEARCH），而非重复 tool
  2. search_success_rate 从 state.failed_searches 计算，而非硬编码 0
  3. n_failed 从 episode_result.trajectory 中的 SEARCH 结果计算
"""
from __future__ import annotations
from typing import List, Set, Dict, Any


def compute_citation_metrics(
    cited_ids: List[str],
    ground_truth_ids: List[str],
) -> tuple[float, float, float]:
    """
    计算 citation precision, recall, F1。
    """
    cited_set = set(cited_ids)
    gt_set = set(ground_truth_ids)
    intersection = cited_set & gt_set

    if len(cited_set) == 0:
        precision = 0.0
    else:
        precision = len(intersection) / len(cited_set)

    if len(gt_set) == 0:
        recall = 0.0
    else:
        recall = len(intersection) / len(gt_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_answer_quality(
    answer_text: str,
    ground_truth_answer: str,
    rubric_weights: Dict[str, float],
) -> float:
    """
    计算答案质量分数（当前版本：简单关键词覆盖）。
    """
    if not answer_text:
        return 0.0

    if ground_truth_answer:
        gt_keywords = set(ground_truth_answer.lower().split())
        answer_keywords = set(answer_text.lower().split())
        overlap = gt_keywords & answer_keywords
        keyword_coverage = len(overlap) / max(len(gt_keywords), 1)
    else:
        keyword_coverage = 0.5

    sentences = answer_text.split(".")
    structural = min(len([s for s in sentences if len(s.strip()) > 20]) / 3.0, 1.0)

    max_reasonable_length = 2000
    no_hallucination = min(len(answer_text) / max_reasonable_length, 1.0)

    score = (
        keyword_coverage * rubric_weights.get("keyword_coverage", 0.4)
        + structural * rubric_weights.get("structural_completeness", 0.3)
        + no_hallucination * rubric_weights.get("no_hallucination", 0.3)
    )
    return min(max(score, 0.0), 1.0)


def compute_task_success(
    answer_quality: float,
    citation_recall: float,
    min_answer_quality: float = 0.5,
    min_citation_recall: float = 0.3,
) -> bool:
    """
    判断任务是否成功。
    """
    return answer_quality >= min_answer_quality and citation_recall >= min_citation_recall


def compute_process_metrics(
    trajectory: List[Any],
    invalid_action_count: int,
    search_history: List[Dict[str, Any]],
    failed_searches: List[str],
    cited_chunks: List[str],
    ground_truth_citations: List[str],
) -> Dict[str, float]:
    """
    计算过程级指标。

    修复:
      - repeated_query_rate: 计算重复 query（同一 query 多次 SEARCH），而非重复 tool
      - search_success_rate: 从 failed_searches 计算，而非硬编码

    指标：
    - repeated_query_rate: 重复 query 所占比例
    - invalid_action_rate: 无效 action 所占比例
    - search_success_rate: 搜索成功率
    - evidence_coverage: cited chunks 覆盖 ground truth 的比例
    - tool_diversity: 使用的不同 tool 类型数 / 5
    - answer_conciseness: answer 长度 / 合理上限
    """
    total_steps = len(trajectory)
    if total_steps == 0:
        return {}

    # 修复: repeated_query_rate = 重复 query / 总 search 次数
    # 从 search_history 中提取所有 query
    search_queries = [h["query"] for h in search_history if "query" in h]
    if len(search_queries) > 0:
        unique_queries = set(search_queries)
        repeated_searches = len(search_queries) - len(unique_queries)
        repeated_query_rate = repeated_searches / max(len(search_queries), 1)
    else:
        repeated_query_rate = 0.0

    # invalid_action_rate
    invalid_action_rate = invalid_action_count / total_steps

    # 修复: search_success_rate = (总搜索数 - 失败搜索数) / 总搜索数
    n_searches = len(search_queries)
    n_failed = len(failed_searches)
    if n_searches > 0:
        search_success_rate = (n_searches - n_failed) / n_searches
    else:
        search_success_rate = 1.0  # 无搜索时视为成功

    # evidence_coverage
    cited_set = set(cited_chunks)
    gt_set = set(ground_truth_citations)
    if len(gt_set) > 0:
        evidence_coverage = len(cited_set & gt_set) / len(gt_set)
    else:
        evidence_coverage = 0.0

    # tool_diversity
    tool_names = []
    for step in trajectory:
        if hasattr(step, "tool_name"):
            tool_names.append(step.tool_name)
        elif hasattr(step, "tool"):
            tool_names.append(step.tool)
    used_tools = set(tool_names)
    tool_diversity = len(used_tools) / 5.0

    return {
        "repeated_query_rate": repeated_query_rate,
        "invalid_action_rate": invalid_action_rate,
        "search_success_rate": search_success_rate,
        "evidence_coverage": evidence_coverage,
        "tool_diversity": tool_diversity,
    }
