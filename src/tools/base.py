"""
src/tools/base.py
职责: BaseTool 抽象基类 + ToolResult 统一返回结构
设计: execute(state) 接收 EnvState；返回 ToolResult；统一序列化接口
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """
    工具执行结果的统一封装。

    设计理由：
    - 区分 success 和 error，避免异常滥用
    - data 字段承载工具特定输出
    - stats 字段携带执行统计（n_candidates, evidence_gain 等）
    """
    success: bool
    data: Any = None
    error: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """生成简短描述，用于 Observation.last_tool_result_summary。"""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, list):
            return f"Returned {len(self.data)} items"
        if isinstance(self.data, dict):
            keys = list(self.data.keys())[:3]
            return f"Keys: {keys}"
        return str(self.data)[:200]

    def is_empty(self) -> bool:
        """判断是否为空结果。"""
        if not self.success:
            return True
        if self.data is None:
            return True
        if isinstance(self.data, list) and len(self.data) == 0:
            return True
        return False

    def to_dict(self) -> dict:
        """统一序列化格式。"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "stats": self.stats,
        }


class BaseTool(ABC):
    """
    工具抽象基类。

    所有 5 个原子动作（SEARCH/READ/RERANK/CITE/ANSWER）都继承此基类。
    """
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, params: dict, state: Any) -> ToolResult:
        """
        执行工具逻辑。

        Args:
            params: Action.params
            state: EnvState（可通过 state.get_corpus() 获取 corpus）

        Returns:
            ToolResult
        """
        raise NotImplementedError

    def validate_params(self, params: dict) -> tuple[bool, str]:
        """
        参数校验。默认不做校验，子类可重写。
        """
        return True, ""

    def safe_execute(self, params: dict, state: Any) -> ToolResult:
        """带错误处理的执行包装。"""
        is_valid, error = self.validate_params(params)
        if not is_valid:
            return ToolResult(success=False, error=error)
        try:
            return self.execute(params, state)
        except Exception as e:
            return ToolResult(success=False, error=f"{self.name} failed: {e}")
