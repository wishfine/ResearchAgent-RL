"""
src/environment/state.py
职责: EnvState——Environment 内部状态管理
设计: 持有 task、corpus 引用（_corpus）、所有步级计数器；提供状态更新接口
修复: 确保 _corpus 字段存在，工具可通过 state._corpus 访问
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..schema.document import CandidateChunk, ReadSummary, Chunk
from ..schema.task import TaskSample
from ..schema.action import Action, TrajectoryStep


@dataclass
class EnvState:
    """
    Environment 内部状态。
    所有字段都是 mutable 的，每次 step 后更新。
    """
    task: TaskSample
    _corpus: Any = None  # CorpusStore 引用，由 env.reset() 注入
    # 内部计数器
    current_step: int = 0
    remaining_steps: int = 15
    max_steps: int = 15
    max_invalid_actions: int = 3
    max_no_progress: int = 3
    # 动作计数
    invalid_action_count: int = 0
    no_progress_count: int = 0
    repeated_action_count: int = 0
    # 检索结果
    candidate_chunks: List[CandidateChunk] = field(default_factory=list)
    read_summaries: List[ReadSummary] = field(default_factory=list)
    cited_chunks: List[str] = field(default_factory=list)
    # 搜索历史
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    failed_searches: List[str] = field(default_factory=list)
    # 轨迹
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    # 终止
    done_reason: Optional[str] = None
    final_answer: Optional[str] = None
    # 缓存上次状态（用于检测 no_progress）
    _prev_n_candidates: int = 0
    _prev_n_read: int = 0
    _prev_n_cited: int = 0

    def record_step(self, action: Action, tool_result: Any, is_valid: bool, error: str = "") -> None:
        """记录一步执行。"""
        step = TrajectoryStep(
            step_idx=self.current_step,
            action=action,
            tool_name=action.tool,
            tool_params=action.params,
            tool_result=tool_result,
            is_valid=is_valid,
            error_message=error,
        )
        self.trajectory.append(step)
        self.current_step += 1
        self.remaining_steps -= 1

    def record_invalid_action(self, error: str) -> None:
        self.invalid_action_count += 1

    def check_no_progress(self) -> None:
        """
        检测是否无实质进展。
        条件：candidate_chunks、read_summaries、cited_chunks 均无新增。
        """
        curr_n_candidates = len(self.candidate_chunks)
        curr_n_read = len(self.read_summaries)
        curr_n_cited = len(self.cited_chunks)

        no_new_candidates = (curr_n_candidates == self._prev_n_candidates)
        no_new_reads = (curr_n_read == self._prev_n_read)
        no_new_cites = (curr_n_cited == self._prev_n_cited)

        if no_new_candidates and no_new_reads and no_new_cites:
            self.no_progress_count += 1
        else:
            self.no_progress_count = 0

        self._prev_n_candidates = curr_n_candidates
        self._prev_n_read = curr_n_read
        self._prev_n_cited = curr_n_cited

    def check_repeated_action(self) -> None:
        """
        检测连续重复动作（同一 tool）。
        更新 repeated_action_count。
        """
        if len(self.trajectory) >= 2:
            last_two = self.trajectory[-2:]
            if last_two[0].tool_name == last_two[1].tool_name:
                self.repeated_action_count += 1

    def is_terminated(self) -> tuple[bool, Optional[str]]:
        """检查是否满足终止条件。返回 (terminated, reason)。"""
        if self.remaining_steps <= 0:
            return True, "max_steps"
        if self.invalid_action_count >= self.max_invalid_actions:
            return True, "invalid_actions_exceeded"
        if self.no_progress_count >= self.max_no_progress:
            return True, "no_progress_exceeded"
        if self.final_answer is not None:
            return True, "answer_submitted"
        return False, None

    def add_candidates(self, candidates: List[CandidateChunk]) -> List[CandidateChunk]:
        """追加候选 chunks（去重）。返回实际新增的 candidates。"""
        existing_ids = {c.chunk_id for c in self.candidate_chunks}
        new_candidates = []
        for c in candidates:
            if c.chunk_id not in existing_ids:
                self.candidate_chunks.append(c)
                existing_ids.add(c.chunk_id)
                new_candidates.append(c)
        return new_candidates

    def add_read_summary(self, summary: ReadSummary) -> bool:
        """追加阅读摘要（去重）。返回是否实际新增。"""
        existing_ids = {s.chunk_id for s in self.read_summaries}
        if summary.chunk_id not in existing_ids:
            self.read_summaries.append(summary)
            return True
        return False

    def cite_chunks(self, chunk_ids: List[str]) -> List[str]:
        """引用 chunks（去重）。返回实际新增的 chunk_ids。"""
        new_cited = []
        for cid in chunk_ids:
            if cid not in self.cited_chunks:
                self.cited_chunks.append(cid)
                new_cited.append(cid)
        return new_cited

    def add_failed_search(self, query: str) -> None:
        if query not in self.failed_searches:
            self.failed_searches.append(query)

    def set_done_reason(self, reason: str) -> None:
        """设置终止原因。"""
        self.done_reason = reason

    def get_corpus(self):
        """获取 corpus 引用。"""
        return self._corpus

    def set_corpus(self, corpus) -> None:
        """设置 corpus 引用。"""
        self._corpus = corpus
