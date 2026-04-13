"""
ResearchAgent-RL Core Data Schemas
===================================
定义项目中所有的核心数据结构，采用 Python dataclass 和 Pydantic 兼容设计。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


# =============================================================================
# 基础数据类型
# =============================================================================

class TaskType(Enum):
    SURVEY = "survey"
    COMPARISON = "comparison"
    EXPERIMENT_DESIGN = "experiment_design"


class ToolName(Enum):
    SEARCH = "SEARCH"
    READ = "READ"
    RERANK = "RERANK"
    CITE = "CITE"
    ANSWER = "ANSWER"


class TerminationType(Enum):
    ANSWER = "answer"
    MAX_STEPS = "max_steps"
    INVALID = "invalid"
    NO_PROGRESS = "no_progress"


# =============================================================================
# 文档与 Chunk 相关
# =============================================================================

@dataclass
class Chunk:
    """文档块，是检索和引用的基本单元"""
    chunk_id: str
    doc_id: str
    content: str                    # chunk 原文（完整内容）
    summary: Optional[str] = None   # 可选的摘要
    title: Optional[str] = None     # 所属文档标题
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """完整文档，由多个 chunk 组成"""
    doc_id: str
    title: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 任务相关
# =============================================================================

@dataclass
class Rubric:
    """评分 rubric，定义任务的质量标准"""
    required_keywords: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)
    optional_keywords: List[str] = field(default_factory=list)
    answer_template: Optional[str] = None


@dataclass
class TaskSample:
    """单条任务样本"""
    task_id: str
    task_type: TaskType
    user_query: str                  # 用户原始问题
    gold_chunks: List[str] = field(default_factory=list)  # 正确答案引用的 chunk_ids
    rubric: Optional[Rubric] = None
    reference_answer: Optional[str] = None  # 参考答案（用于评测）


# =============================================================================
# 动作与工具调用
# =============================================================================

@dataclass
class SearchParams:
    """SEARCH 动作参数"""
    query: str
    topk: int = 5
    target_gap: str = ""            # 当前证据缺口描述
    query_rationale: str = ""       # 为什么要这样检索


@dataclass
class ReadParams:
    """READ 动作参数"""
    chunk_ids: List[str]
    read_goal: str = ""             # 这次阅读要解决什么问题


@dataclass
class RerankParams:
    """RERANK 动作参数"""
    query: str
    candidate_chunk_ids: List[str]
    topk: int = 5
    rerank_goal: str = ""


@dataclass
class CiteParams:
    """CITE 动作参数"""
    chunk_ids: List[str]
    claims: List[str]               # 每个 chunk 对应的 claim


@dataclass
class AnswerParams:
    """ANSWER 动作参数"""
    answer_text: str
    cited_chunk_ids: List[str]


@dataclass
class Action:
    """
    统一 Action Schema
    采用"高层选工具 + 低层生成参数"的设计
    """
    tool: ToolName
    intent: str                      # 动作意图描述
    params: Dict[str, Any]           # 参数字典（根据 tool 类型动态解析）
    reasoning: Optional[str] = None  # 思维链（可选）

    def get_search_params(self) -> SearchParams:
        p = self.params
        return SearchParams(
            query=p.get("query", ""),
            topk=p.get("topk", 5),
            target_gap=p.get("target_gap", ""),
            query_rationale=p.get("query_rationale", "")
        )

    def get_read_params(self) -> ReadParams:
        p = self.params
        return ReadParams(
            chunk_ids=p.get("chunk_ids", []),
            read_goal=p.get("read_goal", "")
        )

    def get_rerank_params(self) -> RerankParams:
        p = self.params
        return RerankParams(
            query=p.get("query", ""),
            candidate_chunk_ids=p.get("candidate_chunk_ids", []),
            topk=p.get("topk", 5),
            rerank_goal=p.get("rerank_goal", "")
        )

    def get_cite_params(self) -> CiteParams:
        p = self.params
        return CiteParams(
            chunk_ids=p.get("chunk_ids", []),
            claims=p.get("claims", [])
        )

    def get_answer_params(self) -> AnswerParams:
        p = self.params
        return AnswerParams(
            answer_text=p.get("answer_text", ""),
            cited_chunk_ids=p.get("cited_chunk_ids", [])
        )


# =============================================================================
# Observation 相关
# =============================================================================

@dataclass
class CandidateChunk:
    """候选 chunk，用于检索结果展示"""
    chunk_id: str
    doc_id: str
    score: float = 0.0
    snippet: str = ""               # 用于展示的片段


@dataclass
class ReadSummary:
    """已读 chunk 的摘要"""
    chunk_id: str
    doc_id: str
    summary: str                     # 阅读后生成的摘要
    snippet: str = ""               # 关键片段
    has_new_evidence: bool = False   # 是否提供了新证据


@dataclass
class StepRecord:
    """轨迹中的单步记录"""
    step_idx: int
    tool: ToolName
    action: Action
    tool_result: str                 # 工具执行结果（字符串形式）
    step_reward: float = 0.0


@dataclass
class Observation:
    """
    Agent 看到的完整 Observation
    采用结构化字段 + prompt serialization，不使用纯 rolling context
    """
    task_id: str
    task_type: TaskType
    user_query: str

    current_subgoal: str = ""        # 当前这一步要解决的子问题
    current_evidence_gap: str = ""   # 当前缺什么证据

    candidate_chunks: List[CandidateChunk] = field(default_factory=list)
    read_summaries: List[ReadSummary] = field(default_factory=list)
    cited_chunks: List[str] = field(default_factory=list)

    search_history: List[str] = field(default_factory=list)
    failed_searches: List[str] = field(default_factory=list)

    trajectory: List[StepRecord] = field(default_factory=list)
    last_tool_result: Optional[str] = None

    remaining_steps: int = 20

    invalid_action_count: int = 0
    repeated_action_count: int = 0
    no_progress_count: int = 0

    final_answer: Optional[str] = None

    def to_prompt_string(self) -> str:
        """序列化为 prompt 字符串，供 policy model 输入"""
        lines = [
            f"Task: {self.user_query}",
            f"Task Type: {self.task_type.value}",
            f"Remaining Steps: {self.remaining_steps}",
            f"",
            f"Current Subgoal: {self.current_subgoal}",
            f"Current Evidence Gap: {self.current_evidence_gap}",
            f"",
        ]

        if self.candidate_chunks:
            lines.append("=== Retrieved Chunks ===")
            for c in self.candidate_chunks[:10]:  # 限制数量避免 context 过长
                lines.append(f"[{c.chunk_id}] score={c.score:.3f}")
                lines.append(c.snippet[:200])
                lines.append("")
            lines.append("")

        if self.read_summaries:
            lines.append("=== Read Summaries ===")
            for s in self.read_summaries:
                lines.append(f"[{s.chunk_id}] {s.summary[:150]}")
            lines.append("")

        if self.cited_chunks:
            lines.append(f"Cited Chunks: {', '.join(self.cited_chunks)}")
            lines.append("")

        if self.search_history:
            lines.append(f"Search History: {' | '.join(self.search_history)}")
            lines.append("")

        if self.last_tool_result:
            lines.append("=== Last Tool Result ===")
            lines.append(self.last_tool_result[:500])
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Reward 相关
# =============================================================================

@dataclass
class RewardBreakdown:
    """Reward 分解"""
    r_final: float = 0.0
    r_citation: float = 0.0
    r_step_sum: float = 0.0
    r_total: float = 0.0


@dataclass
class StepRewardInfo:
    """单步 reward 详情"""
    step_idx: int
    tool: ToolName
    reward: float
    reason: str


# =============================================================================
# 轨迹与回放相关
# =============================================================================

@dataclass
class TrajectoryStep:
    """轨迹中的一步（用于存储和回放）"""
    obs: Dict[str, Any]              # Observation 的字典形式
    action: Dict[str, Any]          # Action 的字典形式
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """完整 episode 的结果"""
    task_id: str
    termination: TerminationType
    total_reward: float
    reward_breakdown: RewardBreakdown
    final_answer: Optional[str]
    cited_chunks: List[str]
    trajectory: List[TrajectoryStep]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ReplayEntry:
    """回放缓冲区中的单条记录"""
    obs: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_obs: Optional[Dict[str, Any]]
    done: bool
    priority: float = 1.0           # 优先级（可用于 PER）


@dataclass
class TrajectoryReplayBuffer:
    """轨迹回放缓冲区"""
    entries: List[ReplayEntry] = field(default_factory=list)
    max_size: int = 10000

    def add(self, entry: ReplayEntry):
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def sample(self, batch_size: int) -> List[ReplayEntry]:
        import random
        return random.sample(self.entries, min(batch_size, len(self.entries)))

    def __len__(self):
        return len(self.entries)
