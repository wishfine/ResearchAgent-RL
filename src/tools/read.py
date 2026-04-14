"""
src/tools/read.py
职责: 实现 READ 工具
设计: 从 corpus 获取 chunk 内容；提取关键词作为 summary；更新 state.read_summaries
修复: evidence_gain 基于新增的 read_summary 计算
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseTool, ToolResult
from ..environment.state import EnvState
from ..schema.document import ReadSummary


class ReadTool(BaseTool):
    name = "READ"
    description = "Read chunks and extract key claims and evidence"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        """
        READ 执行逻辑。

        params:
            chunk_ids: List[str]    要读取的 chunk IDs
            read_goal: str          读取目标描述

        更新 state:
            - read_summaries: 追加新的摘要（去重）

        修复: evidence_gain 基于实际新增的 read_summary 计算
        """
        chunk_ids = params["chunk_ids"]
        read_goal = params.get("read_goal", "extract key claims and evidence")

        if not chunk_ids:
            return ToolResult(success=False, error="chunk_ids is empty")

        corpus = state.get_corpus()
        if corpus is None:
            return ToolResult(success=False, error="Corpus not available")

        chunks = corpus.get_chunks(chunk_ids)
        if not chunks:
            return ToolResult(success=False, error=f"No chunks found for IDs: {chunk_ids}")

        summaries = []
        prev_n_read = len(state.read_summaries)

        for chunk in chunks:
            summary = self._simple_extract(chunk, read_goal)
            # 修复: 使用 add_read_summary() 的返回值判断是否实际新增
            is_new = state.add_read_summary(summary)
            summaries.append({
                "chunk_id": chunk.chunk_id,
                "summary": summary.summary,
                "n_claims": len(summary.key_claims),
                "is_new": is_new,
            })

        # 修复: evidence_gain 基于实际新增的 read_summary 计算
        n_actually_new = len([s for s in summaries if s["is_new"]])
        new_summaries = state.read_summaries[prev_n_read:]
        evidence_gain = sum(s.evidence_strength for s in new_summaries) / max(len(new_summaries), 1)

        return ToolResult(
            success=True,
            data={"summaries": summaries, "read_goal": read_goal},
            stats={
                "n_read": len(chunks),
                "n_new": n_actually_new,
                "evidence_gain": evidence_gain,
            }
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "chunk_ids" not in params:
            return False, "READ requires 'chunk_ids' param"
        if not isinstance(params["chunk_ids"], list):
            return False, "READ 'chunk_ids' must be list"
        return True, ""

    def _simple_extract(self, chunk, read_goal: str) -> ReadSummary:
        """
        简单关键词提取（占位实现）。
        后续替换为 LLM summarization。
        """
        content = chunk.content
        summary_text = content[:200] + ("..." if len(content) > 200 else "")

        keywords = ["method", "approach", "result", "experiment", "model", "training", "performance"]
        sentences = content.split(".")
        key_claims = [
            s.strip() for s in sentences
            if any(kw in s.lower() for kw in keywords) and len(s.strip()) > 20
        ][:5]

        evidence_strength = min(len(key_claims) / 3.0, 1.0)

        return ReadSummary(
            chunk_id=chunk.chunk_id,
            summary=summary_text,
            key_claims=key_claims,
            evidence_strength=evidence_strength,
            key_passages=[content[:300]],
        )
