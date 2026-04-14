"""
src/tools/cite.py
职责: 实现 CITE 工具
设计: 将 chunk_ids 标记为已引用；检查 claim 和 chunk 的对应关系
修复: cite_chunks() 返回实际新增的 chunk_ids
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseTool, ToolResult
from ..environment.state import EnvState


class CiteTool(BaseTool):
    name = "CITE"
    description = "Cite chunks to support specific claims"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        """
        CITE 执行逻辑。

        params:
            chunk_ids: List[str]    要引用的 chunk IDs
            claims: List[str]        要支持的 claims

        更新 state:
            - cited_chunks: 追加新的 citation（去重）

        修复: 使用 cite_chunks() 的返回值计算 n_new
        """
        chunk_ids = params["chunk_ids"]
        claims = params["claims"]

        if not chunk_ids:
            return ToolResult(success=False, error="chunk_ids is empty")
        if not claims:
            return ToolResult(success=False, error="claims is empty")

        warnings = []

        # 检查是否有未读 chunk 被 cite
        read_chunk_ids = {s.chunk_id for s in state.read_summaries}
        unread_cited = [cid for cid in chunk_ids if cid not in read_chunk_ids]
        if unread_cited:
            warnings.append(f"Warning: {len(unread_cited)} chunks cited without being read first")

        # 检查 claim 数量与 chunk 数量
        if len(claims) != len(chunk_ids):
            warnings.append(f"Warning: {len(claims)} claims for {len(chunk_ids)} chunks")

        # 修复: cite_chunks() 返回实际新增的 chunk_ids
        prev_n_cited = len(state.cited_chunks)
        new_cited = state.cite_chunks(chunk_ids)
        n_new = len(new_cited)

        return ToolResult(
            success=True,
            data={
                "cited_chunk_ids": chunk_ids,
                "n_cited_total": len(state.cited_chunks),
                "n_new": n_new,
                "warnings": warnings,
            },
            stats={
                "n_cited": len(chunk_ids),
                "n_new": n_new,
            }
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "chunk_ids" not in params or "claims" not in params:
            return False, "CITE requires both 'chunk_ids' and 'claims'"
        return True, ""
