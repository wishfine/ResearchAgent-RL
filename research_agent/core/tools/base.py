from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
from abc import ABC, abstractmethod

@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, list):
            return f"Returned {len(self.data)} items"
        if isinstance(self.data, dict):
            keys = list(self.data.keys())[:3]
            return f"Keys: {keys}"
        return str(self.data)[:200]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "stats": self.stats,
        }


class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, params: dict, state: Any) -> ToolResult:
        raise NotImplementedError

    def validate_params(self, params: dict) -> tuple[bool, str]:
        return True, ""
