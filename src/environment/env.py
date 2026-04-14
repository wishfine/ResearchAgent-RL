"""
src/environment/env.py
职责: ResearchEnv——核心 Environment 主类
设计: reset/step/finalize_episode 三段式; 持有 CorpusStore; 构建 Observation 暴露给 Agent
修复:
  1. reset() 中注入 corpus: state.set_corpus(self.corpus)
  2. step() 中正确设置 done_reason: state.set_done_reason(reason)
  3. finalize_episode() 使用 state.done_reason 而非从外部传入
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..schema.task import TaskSample
from ..schema.observation import Observation
from ..schema.action import Action
from ..schema.result import EpisodeResult
from ..tools.base import BaseTool, ToolResult
from .state import EnvState
from .corpus import CorpusStore


class ResearchEnv:
    """
    核心 Environment。

    使用方式:
        env = ResearchEnv(corpus, max_steps=15)
        env.register_tool(SearchTool())
        env.register_tool(ReadTool())
        ...
        obs = env.reset(task)
        for _ in range(15):
            action = policy.decide(obs)
            obs, done, reason = env.step(action)
            if done:
                break
        result = env.finalize_episode()
    """

    def __init__(
        self,
        corpus: CorpusStore,
        max_steps: int = 15,
        max_invalid_actions: int = 3,
        max_no_progress: int = 3,
    ):
        self.corpus = corpus
        self.max_steps = max_steps
        self.max_invalid_actions = max_invalid_actions
        self.max_no_progress = max_no_progress
        self._state: Optional[EnvState] = None
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具。工具将收到带有 corpus 引用的 state。"""
        self._tools[tool.name] = tool

    def reset(self, task: TaskSample) -> Observation:
        """
        初始化 episode。
        返回第一个 Observation。

        修复: 在 state 中注入 corpus 引用，工具可通过 state._corpus 访问
        """
        self._state = EnvState(
            task=task,
            remaining_steps=self.max_steps,
            max_steps=self.max_steps,
            max_invalid_actions=self.max_invalid_actions,
            max_no_progress=self.max_no_progress,
        )
        # 修复1: 注入 corpus 到 state
        self._state.set_corpus(self.corpus)

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, bool, str]:
        """
        执行一个 action。

        修复: 正确设置 done_reason 到 state 中

        Returns:
            observation: 执行后的状态视图
            is_done: 是否终止
            done_reason: 终止原因
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        state = self._state

        # 1. 校验 action 合法性
        is_valid, error_msg = action.validate()
        if not is_valid:
            state.record_invalid_action(error_msg)
            state.record_step(action, None, is_valid=False, error=error_msg)
            terminated, reason = state.is_terminated()
            # 修复2: 正确设置 done_reason
            if reason is not None:
                state.set_done_reason(reason)
            return self._build_observation(), terminated, reason or "invalid_action"

        # 2. 获取工具并执行
        tool = self._tools.get(action.tool)
        if tool is None:
            error_msg = f"Tool {action.tool} not registered"
            state.record_invalid_action(error_msg)
            state.record_step(action, None, is_valid=False, error=error_msg)
            terminated, reason = state.is_terminated()
            if reason is not None:
                state.set_done_reason(reason)
            return self._build_observation(), terminated, reason or "unknown_tool"

        try:
            result = tool.execute(action.params, state)
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            state.record_invalid_action(error_msg)
            state.record_step(action, None, is_valid=False, error=error_msg)
            terminated, reason = state.is_terminated()
            if reason is not None:
                state.set_done_reason(reason)
            return self._build_observation(), terminated, reason or "tool_error"

        # 3. 记录有效步骤
        state.record_step(action, result, is_valid=True)

        # 4. 检查 no_progress 和 repeated_action
        state.check_no_progress()
        state.check_repeated_action()

        # 5. 检查终止条件
        terminated, reason = state.is_terminated()
        # 修复2: 正确设置 done_reason
        if reason is not None:
            state.set_done_reason(reason)

        return self._build_observation(), terminated, reason or "unknown"

    def finalize_episode(self, final_answer: Optional[str] = None) -> EpisodeResult:
        """
        结束 episode，返回完整记录。
        使用 state.done_reason（已在 step() 中正确设置）。
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before finalize_episode()")

        state = self._state
        if final_answer is not None:
            state.final_answer = final_answer
        elif state.final_answer is None:
            state.final_answer = ""

        episode_result = EpisodeResult(
            task_id=state.task.task_id,
            trajectory=state.trajectory,
            final_answer=state.final_answer,
            cited_chunk_ids=state.cited_chunks.copy(),
            # 修复: 使用 state.done_reason
            done_reason=state.done_reason or "max_steps",
            total_steps=state.current_step,
            n_valid_steps=sum(1 for s in state.trajectory if s.is_valid),
            n_invalid_steps=sum(1 for s in state.trajectory if not s.is_valid),
        )
        return episode_result

    def submit_answer(self, answer_text: str, cited_chunk_ids: list) -> tuple[Observation, bool, str]:
        """快捷方法：直接提交 ANSWER action。"""
        action = Action.answer(
            answer_text=answer_text,
            cited_chunk_ids=cited_chunk_ids,
        )
        if self._state is not None:
            self._state.final_answer = answer_text
            self._state.cite_chunks(cited_chunk_ids)
        return self.step(action)

    def _build_observation(self) -> Observation:
        """根据当前 state 构建 Observation。"""
        state = self._state
        task = state.task

        trajectory_tools = [s.tool_name for s in state.trajectory]
        last_tool = trajectory_tools[-1] if trajectory_tools else None

        last_result_summary = ""
        if state.trajectory:
            last_step = state.trajectory[-1]
            if last_step.tool_result is not None:
                if isinstance(last_step.tool_result, ToolResult):
                    last_result_summary = last_step.tool_result.summary()
                elif hasattr(last_step.tool_result, "summary"):
                    last_result_summary = last_step.tool_result.summary()
                else:
                    last_result_summary = str(last_step.tool_result)[:200]

        return Observation(
            task_id=task.task_id,
            task_type=task.task_type.value,
            user_query=task.user_query,
            current_subgoal=self._infer_subgoal(state),
            current_evidence_gap=self._infer_evidence_gap(state),
            candidate_chunks=state.candidate_chunks.copy(),
            read_summaries=state.read_summaries.copy(),
            cited_chunks=state.cited_chunks.copy(),
            search_history=state.search_history.copy(),
            failed_searches=state.failed_searches.copy(),
            trajectory=trajectory_tools,
            last_tool=last_tool,
            last_tool_result_summary=last_result_summary,
            remaining_steps=state.remaining_steps,
            invalid_action_count=state.invalid_action_count,
            repeated_action_count=state.repeated_action_count,
            no_progress_count=state.no_progress_count,
            final_answer=state.final_answer,
        )

    def _infer_subgoal(self, state: EnvState) -> str:
        """根据当前状态推断子目标。"""
        if state.remaining_steps >= state.max_steps - 2:
            return "理解问题，制定检索策略"
        if len(state.candidate_chunks) == 0:
            return "执行首次检索，找到相关文档"
        if len(state.read_summaries) < min(len(state.candidate_chunks), 3):
            return "阅读候选文档，提取关键证据"
        if len(state.cited_chunks) < 3:
            return "整理证据，引用支持的 chunk"
        return "综合证据，形成最终答案"

    def _infer_evidence_gap(self, state: EnvState) -> str:
        """根据当前状态推断证据缺口。"""
        if len(state.candidate_chunks) == 0:
            return "尚无候选文档，需要检索"
        if len(state.read_summaries) == 0:
            return "有候选但未阅读，需要阅读提取证据"
        if len(state.cited_chunks) == 0:
            return "已阅读但未引用，需要整理证据"
        unread = [c.chunk_id for c in state.candidate_chunks
                  if c.chunk_id not in {s.chunk_id for s in state.read_summaries}]
        if unread:
            return f"还有 {len(unread)} 个候选 chunk 未阅读"
        return "证据已充分收集，可以作答"
