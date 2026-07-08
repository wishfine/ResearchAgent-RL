from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass

from research_agent.core.schema.task import TaskSample
from research_agent.core.schema.observation import Observation
from research_agent.core.schema.action import Action
from research_agent.core.schema.result import EpisodeResult
from research_agent.core.tools.base import BaseTool, ToolResult
from .state import EnvState
from research_agent.core.corpus.store import CorpusStore

class ResearchEnv:
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
        self._tools[tool.name] = tool

    def reset(self, task: TaskSample) -> Observation:
        self._state = EnvState(
            task=task,
            remaining_steps=self.max_steps,
            max_steps=self.max_steps,
            max_invalid_actions=self.max_invalid_actions,
            max_no_progress=self.max_no_progress,
        )
        self._state.set_corpus(self.corpus)
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, bool, str]:
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        state = self._state

        # 1. Validate action parameters and schema
        is_valid, error_msg = action.validate()
        if not is_valid:
            state.record_invalid_action(error_msg)
            state.record_step(action, None, is_valid=False, error=error_msg)
            terminated, reason = state.is_terminated()
            if reason is not None:
                state.set_done_reason(reason)
            return self._build_observation(), terminated, reason or "invalid_action"

        # 2. Get tool and execute
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

        # 3. Record valid step
        state.record_step(action, result, is_valid=True)

        # 4. Check no progress and repeated action
        state.check_no_progress()
        state.check_repeated_action()

        # 5. Check termination
        terminated, reason = state.is_terminated()
        if reason is not None:
            state.set_done_reason(reason)

        return self._build_observation(), terminated, reason or "unknown"

    def finalize_episode(self, final_answer: Optional[str] = None) -> EpisodeResult:
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
            done_reason=state.done_reason or "max_steps",
            total_steps=state.current_step,
            n_valid_steps=sum(1 for s in state.trajectory if s.is_valid),
            n_invalid_steps=sum(1 for s in state.trajectory if not s.is_valid),
        )
        return episode_result

    def _build_observation(self) -> Observation:
        state = self._state
        task = state.task

        trajectory_tools = [s.tool_name for s in state.trajectory]
        last_tool = trajectory_tools[-1] if trajectory_tools else None

        last_result_summary = ""
        if state.trajectory:
            last_step = state.trajectory[-1]
            if last_step.tool_result is not None:
                if hasattr(last_step.tool_result, "summary"):
                    last_result_summary = last_step.tool_result.summary()
                else:
                    last_result_summary = str(last_step.tool_result)[:200]

        return Observation(
            task_id=task.task_id,
            task_type=task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type),
            user_query=task.user_query,
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
