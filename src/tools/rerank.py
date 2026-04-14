"""
src/tools/rerank.py
职责: 实现 RERANK 工具
设计: 基于 query-chunk 相关性重排；更新 candidate_chunks 的 rank 字段
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseTool, ToolResult
from ..environment.state import EnvState


class RerankTool(BaseTool):
    name = "RERANK"
    description = "Rerank candidate chunks by relevance to query"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        """
        RERANK 执行逻辑。

        params:
            query: str                  检索 query
            candidate_chunk_ids: List[str]  要重排的候选 chunk IDs
            topk: int                   返回 topk
            rerank_goal: str            重排目标描述

        更新 state:
            - candidate_chunks: 更新 rank 字段
        """
        query = params["query"]
        candidate_ids = params["candidate_chunk_ids"]
        topk = params.get("topk", 5)
        rerank_goal = params.get("rerank_goal", "prioritize strongest evidence")

        corpus = state.get_corpus()
        if corpus is None:
            return ToolResult(success=False, error="Corpus not available")

        chunks = corpus.get_chunks(candidate_ids)
        if not chunks:
            return ToolResult(success=False, error=f"No chunks found for IDs")

        scored = []
        for chunk in chunks:
            score = self._compute_relevance(query, chunk, rerank_goal)
            scored.append((chunk.chunk_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        top_ids = [cid for cid, _ in scored[:topk]]

        # 更新 state 中的 candidate_chunks rank
        for candidate in state.candidate_chunks:
            if candidate.chunk_id in top_ids:
                candidate.rank = top_ids.index(candidate.chunk_id) + 1

        return ToolResult(
            success=True,
            data={
                "reranked_ids": top_ids,
                "scores": {cid: score for cid, score in scored[:topk]},
            },
            stats={"n_reranked": len(chunks), "topk": topk}
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        required = ["query", "candidate_chunk_ids", "topk"]
        for k in required:
            if k not in params:
                return False, f"RERANK requires '{k}' param"
        return True, ""

    def _compute_relevance(self, query: str, chunk, rerank_goal: str) -> float:
        """
        计算 query-chunk 相关性（简单版本）。
        """
        query_terms = set(query.lower().split())
        content_lower = chunk.content.lower()

        overlap = sum(1 for term in query_terms if term in content_lower)
        length_norm = 1.0 / (len(chunk.content) / 500 + 1)

        goal_terms = set(rerank_goal.lower().split())
        goal_overlap = sum(1 for term in goal_terms if term in content_lower)

        score = overlap * 1.0 + goal_overlap * 0.5 + length_norm * 0.1
        return score
