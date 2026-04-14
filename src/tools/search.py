"""
src/tools/search.py
职责: 实现 SEARCH 工具
设计: 调用 CorpusStore.search_simple；更新 state.candidate_chunks 和 search_history
修复: n_new 的计算使用 state.add_candidates() 的返回值（实际新增的 candidates）
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseTool, ToolResult
from ..environment.state import EnvState


class SearchTool(BaseTool):
    name = "SEARCH"
    description = "Search corpus for chunks relevant to query"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        """
        SEARCH 执行逻辑。

        params:
            query: str              检索词
            topk: int               返回 topk 候选（默认 10）
            target_gap: str         目标证据缺口描述（可选）
            query_rationale: str    检索理由（可选）

        更新 state:
            - candidate_chunks: 追加新的候选 chunks
            - search_history: 添加本次检索记录
            - failed_searches: 如果返回空，添加 query 到 failed_searches

        修复: n_new 使用 add_candidates() 的返回值计算
        """
        query = params["query"]
        topk = params.get("topk", 10)
        target_gap = params.get("target_gap", "")

        # 获取 corpus
        corpus = state.get_corpus()
        if corpus is None:
            return ToolResult(success=False, error="Corpus not available")

        # 获取允许检索的 doc_ids（任务范围限制）
        task = state.task
        doc_ids = task.reference_docs if task.reference_docs else None

        # 执行检索
        results = corpus.search_simple(query, topk=topk, doc_ids=doc_ids)

        # 修复: n_new 使用 add_candidates() 的返回值计算
        prev_n_candidates = len(state.candidate_chunks)
        new_candidates = state.add_candidates(results)
        n_new = len(new_candidates)

        if results:
            state.search_history.append({
                "query": query,
                "topk": topk,
                "n_results": len(results),
                "n_new": n_new,
                "target_gap": target_gap,
            })
        else:
            state.add_failed_search(query)
            state.search_history.append({
                "query": query,
                "topk": topk,
                "n_results": 0,
                "failed": True,
            })

        return ToolResult(
            success=True,
            data={
                "candidates": [c.to_dict() for c in results],
                "n_candidates": len(results),
                "n_new": n_new,
                "query": query,
            },
            stats={
                "n_candidates": len(results),
                "n_new": n_new,
            }
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "query" not in params:
            return False, "SEARCH requires 'query' param"
        if not isinstance(params.get("topk", 1), int):
            return False, "SEARCH 'topk' must be int"
        return True, ""
