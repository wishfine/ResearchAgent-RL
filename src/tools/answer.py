"""
src/tools/answer.py
职责: 实现 ANSWER 工具
设计: 提交最终答案；标记 episode 结束；自动补全 cited_chunk_ids
"""
from __future__ import annotations
from typing import Any, Dict, List
from .base import BaseTool, ToolResult
from ..environment.state import EnvState


class AnswerTool(BaseTool):
    name = "ANSWER"
    description = "Submit final answer with citations"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        """
        ANSWER 执行逻辑。

        params:
            answer_text: str             最终答案文本
            cited_chunk_ids: List[str]    引用的 chunk IDs

        更新 state:
            - final_answer: 设置最终答案
            - cited_chunks: 自动补全（合并已有的 cited_chunks）

        终止条件：
            - state.final_answer 设置后，is_terminated() 返回 True
        """
        answer_text = params["answer_text"]
        cited_chunk_ids = params.get("cited_chunk_ids", [])

        if not answer_text or len(answer_text.strip()) == 0:
            return ToolResult(success=False, error="answer_text cannot be empty")

        # 自动补全 cited_chunk_ids（合并已有的）
        all_cited = set(state.cited_chunks)
        all_cited.update(cited_chunk_ids)
        all_cited_list = list(all_cited)

        # 验证 cited chunk IDs 是否有效
        corpus = state.get_corpus()
        invalid_ids = []
        if corpus:
            invalid_ids = [cid for cid in all_cited_list if cid not in corpus]

        if invalid_ids:
            return ToolResult(
                success=False,
                error=f"Invalid chunk IDs: {invalid_ids[:5]}",
            )

        # 提交答案
        state.final_answer = answer_text
        state.cited_chunks = all_cited_list

        return ToolResult(
            success=True,
            data={
                "answer_text": answer_text,
                "cited_chunk_ids": all_cited_list,
                "n_citations": len(all_cited_list),
            },
            stats={"answer_length": len(answer_text)}
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "answer_text" not in params:
            return False, "ANSWER requires 'answer_text' param"
        return True, ""
