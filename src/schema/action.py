"""
src/schema/action.py
职责: 定义 Action schema 和 TrajectoryStep
设计: 统一 Action schema 带 tool/intent/params/reasoning; TrajectoryStep 记录每步完整信息
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


VALID_TOOLS = {"SEARCH", "READ", "RERANK", "CITE", "ANSWER"}


@dataclass
class Action:
    """
    统一 Action schema。

    设计理由：
    - tool: 动作类型，显式枚举避免字符串错误
    - intent: 动作意图描述，用于事后分析和 reward shaping
    - params: 工具参数，类型安全
    - reasoning: LLM 推理过程（当前阶段可选，供后续 RL reward shaping 使用）
    """
    tool: str
    intent: str
    params: Dict[str, Any]
    reasoning: Optional[str] = None

    def validate(self) -> tuple[bool, str]:
        """校验 action 合法性。返回 (is_valid, error_message)。"""
        if self.tool not in VALID_TOOLS:
            return False, f"Invalid tool: {self.tool}. Must be one of {VALID_TOOLS}"

        if self.tool == "SEARCH":
            if "query" not in self.params:
                return False, "SEARCH requires 'query' param"
            if not isinstance(self.params.get("topk", 1), int):
                return False, "SEARCH 'topk' must be int"

        elif self.tool == "READ":
            if "chunk_ids" not in self.params:
                return False, "READ requires 'chunk_ids' param"
            if not isinstance(self.params["chunk_ids"], list):
                return False, "READ 'chunk_ids' must be list"

        elif self.tool == "RERANK":
            required = ["query", "candidate_chunk_ids", "topk"]
            for k in required:
                if k not in self.params:
                    return False, f"RERANK requires '{k}' param"

        elif self.tool == "CITE":
            if "chunk_ids" not in self.params or "claims" not in self.params:
                return False, "CITE requires both 'chunk_ids' and 'claims'"

        elif self.tool == "ANSWER":
            if "answer_text" not in self.params:
                return False, "ANSWER requires 'answer_text' param"

        return True, ""

    @classmethod
    def search(
        cls,
        query: str,
        topk: int = 10,
        target_gap: str = "",
        query_rationale: str = "",
        reasoning: Optional[str] = None,
    ) -> "Action":
        return cls(
            tool="SEARCH",
            intent=f"Search for chunks relevant to: {query}",
            params={"query": query, "topk": topk, "target_gap": target_gap, "query_rationale": query_rationale},
            reasoning=reasoning,
        )

    @classmethod
    def read(
        cls,
        chunk_ids: List[str],
        read_goal: str = "extract key claims and evidence",
        reasoning: Optional[str] = None,
    ) -> "Action":
        return cls(
            tool="READ",
            intent=f"Read {len(chunk_ids)} chunks to extract evidence",
            params={"chunk_ids": chunk_ids, "read_goal": read_goal},
            reasoning=reasoning,
        )

    @classmethod
    def rerank(
        cls,
        query: str,
        candidate_chunk_ids: List[str],
        topk: int = 5,
        rerank_goal: str = "prioritize strongest evidence",
        reasoning: Optional[str] = None,
    ) -> "Action":
        return cls(
            tool="RERANK",
            intent=f"Rerank {len(candidate_chunk_ids)} candidates for top {topk}",
            params={"query": query, "candidate_chunk_ids": candidate_chunk_ids, "topk": topk, "rerank_goal": rerank_goal},
            reasoning=reasoning,
        )

    @classmethod
    def cite(
        cls,
        chunk_ids: List[str],
        claims: List[str],
        reasoning: Optional[str] = None,
    ) -> "Action":
        return cls(
            tool="CITE",
            intent=f"Cite {len(chunk_ids)} chunks to support {len(claims)} claims",
            params={"chunk_ids": chunk_ids, "claims": claims},
            reasoning=reasoning,
        )

    @classmethod
    def answer(
        cls,
        answer_text: str,
        cited_chunk_ids: List[str],
        reasoning: Optional[str] = None,
    ) -> "Action":
        return cls(
            tool="ANSWER",
            intent="Submit final answer with citations",
            params={"answer_text": answer_text, "cited_chunk_ids": cited_chunk_ids},
            reasoning=reasoning,
        )

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "intent": self.intent,
            "params": self.params,
            "reasoning": self.reasoning,
        }


@dataclass
class TrajectoryStep:
    """
    轨迹中的单步记录。
    """
    step_idx: int = 0
    action: Action = None
    tool_name: str = ""
    tool_params: dict = field(default_factory=dict)
    tool_result: Any = None
    is_valid: bool = True
    error_message: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        result_data = None
        if self.tool_result is not None:
            if hasattr(self.tool_result, "to_dict"):
                result_data = self.tool_result.to_dict()
            elif isinstance(self.tool_result, dict):
                result_data = self.tool_result
            else:
                result_data = str(self.tool_result)

        return {
            "step_idx": self.step_idx,
            "tool": self.tool_name,
            "params": self.tool_params,
            "intent": self.action.intent,
            "is_valid": self.is_valid,
            "error": self.error_message,
            "result": result_data,
        }
