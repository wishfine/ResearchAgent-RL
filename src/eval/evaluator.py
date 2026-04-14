"""
src/eval/evaluator.py
职责: 主评测器
设计: 输入 EpisodeResult + TaskSample；输出 EvalResult；调用 metrics.py 中的各指标计算函数
修复:
  1. rubric 类型改为 Optional[Rubric]，而非 Optional[TaskSample]
  2. 正确从 episode_result.trajectory 获取 search_history 和 failed_searches
  3. 正确传递 search_history 和 failed_searches 给 compute_process_metrics
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from ..schema.task import TaskSample, Rubric
from ..schema.result import EpisodeResult, EvalResult
from .metrics import (
    compute_citation_metrics,
    compute_answer_quality,
    compute_task_success,
    compute_process_metrics,
)


class Evaluator:
    """
    主评测器。

    使用方式:
        evaluator = Evaluator()
        eval_result = evaluator.evaluate(episode_result, task_sample)
    """

    def evaluate(
        self,
        episode_result: EpisodeResult,
        task_sample: TaskSample,
        rubric: Optional[Rubric] = None,
    ) -> EvalResult:
        """
        执行完整评测。

        修复: rubric 类型为 Optional[Rubric]

        流程：
        1. 计算 citation metrics
        2. 计算 answer quality
        3. 判断 task success
        4. 计算 process metrics
        """
        # 修复: 使用传入的 rubric 或 task_sample.rubric
        actual_rubric = rubric if rubric is not None else task_sample.rubric

        cited_ids = episode_result.cited_chunk_ids
        gt_ids = task_sample.ground_truth_citations
        gt_answer = task_sample.ground_truth_answer or ""

        # 1. citation metrics
        precision, recall, f1 = compute_citation_metrics(cited_ids, gt_ids)

        # 2. answer quality
        answer_quality = compute_answer_quality(
            episode_result.final_answer,
            gt_answer,
            actual_rubric.answer_weights,
        )

        # 3. task success
        task_success = compute_task_success(answer_quality, recall)

        # 4. process metrics - 修复: 正确获取 search_history 和 failed_searches
        search_history, failed_searches = self._extract_search_info(episode_result)

        process = compute_process_metrics(
            trajectory=episode_result.trajectory,
            invalid_action_count=episode_result.n_invalid_steps,
            search_history=search_history,
            failed_searches=failed_searches,
            cited_chunks=cited_ids,
            ground_truth_citations=gt_ids,
        )

        return EvalResult(
            task_id=episode_result.task_id,
            task_success=task_success,
            answer_quality=answer_quality,
            citation_precision=precision,
            citation_recall=recall,
            citation_f1=f1,
            average_steps=float(episode_result.total_steps),
            repeated_query_rate=process.get("repeated_query_rate", 0.0),
            invalid_action_rate=process.get("invalid_action_rate", 0.0),
            search_success_rate=process.get("search_success_rate", 1.0),
            evidence_coverage=process.get("evidence_coverage", 0.0),
            tool_diversity=process.get("tool_diversity", 0.0),
            answer_conciseness=min(len(episode_result.final_answer) / 1500.0, 1.0),
            episode_result=episode_result,
            rubric_scores={},
        )

    def _extract_search_info(self, episode_result: EpisodeResult) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        从 episode_result.trajectory 中提取 search_history 和 failed_searches。

        修复: 从 trajectory 的 step.tool_result 中提取搜索结果
        """
        search_history: List[Dict[str, Any]] = []
        failed_searches: List[str] = []

        for step in episode_result.trajectory:
            if step.tool_name != "SEARCH":
                continue

            tool_result = step.tool_result
            if tool_result is None:
                continue

            # 从 tool_result.data 中提取搜索结果
            if hasattr(tool_result, "data") and tool_result.data:
                data = tool_result.data
                if isinstance(data, dict):
                    # 从 data 中提取 query 和 n_results
                    query = data.get("query", "")
                    n_results = data.get("n_results", 0)
                    n_new = data.get("n_new", 0)

                    search_entry = {
                        "query": query,
                        "n_results": n_results,
                        "n_new": n_new,
                    }

                    if n_results == 0:
                        search_entry["failed"] = True
                        if query and query not in failed_searches:
                            failed_searches.append(query)

                    search_history.append(search_entry)

        return search_history, failed_searches
