"""
ResearchAgent-RL Environment
============================
实现 Research Agent 的 tool-use MDP 环境。

核心接口：
- reset(task) -> Observation
- step(action) -> (Observation, float, bool, dict)
- finalize_episode() -> reward breakdown
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from data.schemas import (
    Observation, Action, TaskSample, TaskType,
    StepRecord, CandidateChunk, ReadSummary,
    TerminationType, RewardBreakdown, StepRewardInfo,
    ToolName, EpisodeResult, TrajectoryStep
)
from tools.base import ToolRegistry, ToolResult, CorpusStore


# =============================================================================
# Action Validation
# =============================================================================

class ActionValidator:
    """动作合法性验证器"""

    MAX_INVALID_ACTIONS = 3
    MAX_NO_PROGRESS = 5

    @classmethod
    def validate(cls, action: Action, obs: Observation, tool_registry: ToolRegistry) -> tuple[bool, Optional[str]]:
        """
        验证 action 是否合法
        返回 (is_valid, error_message)
        """
        # 1. tool 必须在合法工具集合中
        if action.tool not in tool_registry.list_tool_names():
            return False, f"Invalid tool: {action.tool}"

        # 2. remaining_steps 必须 > 0
        if obs.remaining_steps <= 0:
            return False, "No remaining steps"

        # 3. 调用对应工具的 validate
        tool = tool_registry.get_tool(action.tool)
        return tool.validate(action.params)


# =============================================================================
# Transition System
# =============================================================================

class TransitionSystem:
    """
    状态转移系统
    根据 action 更新 Observation
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def transition(self, obs: Observation, action: Action, tool_result: ToolResult) -> Observation:
        """
        根据 action 和 tool_result 更新 observation
        """
        # 创建 obs 副本（避免原地修改）
        new_obs = Observation(
            task_id=obs.task_id,
            task_type=obs.task_type,
            user_query=obs.user_query,
            remaining_steps=obs.remaining_steps - 1,
            invalid_action_count=obs.invalid_action_count,
            repeated_action_count=obs.repeated_action_count,
            no_progress_count=obs.no_progress_count,
        )

        # 复制已有数据
        new_obs.candidate_chunks = list(obs.candidate_chunks)
        new_obs.read_summaries = list(obs.read_summaries)
        new_obs.cited_chunks = list(obs.cited_chunks)
        new_obs.search_history = list(obs.search_history)
        new_obs.failed_searches = list(obs.failed_searches)
        new_obs.trajectory = list(obs.trajectory)
        new_obs.current_subgoal = obs.current_subgoal
        new_obs.current_evidence_gap = obs.current_evidence_gap

        # 记录 step
        step_record = StepRecord(
            step_idx=len(obs.trajectory),
            tool=action.tool,
            action=action,
            tool_result=str(tool_result.data) if tool_result.data else tool_result.error or "",
            step_reward=0.0  # step reward 后面由 reward_computer 计算
        )
        new_obs.trajectory.append(step_record)
        new_obs.last_tool_result = str(tool_result.data)[:500] if tool_result.data else None

        # 根据 action.tool 执行不同的状态更新逻辑
        if action.tool == ToolName.SEARCH:
            self._update_for_search(new_obs, action, tool_result)
        elif action.tool == ToolName.READ:
            self._update_for_read(new_obs, action, tool_result)
        elif action.tool == ToolName.RERANK:
            self._update_for_rerank(new_obs, action, tool_result)
        elif action.tool == ToolName.CITE:
            self._update_for_cite(new_obs, action, tool_result)
        elif action.tool == ToolName.ANSWER:
            self._update_for_answer(new_obs, action, tool_result)

        return new_obs

    def _update_for_search(self, obs: Observation, action: Action, result: ToolResult):
        """SEARCH 动作的状态更新"""
        search_params = action.get_search_params()

        # 记录 search_history
        obs.search_history.append(search_params.query)

        if result.success and result.data:
            new_candidates = result.data  # List[CandidateChunk]

            # 去重追加
            existing_ids = {c.chunk_id for c in obs.candidate_chunks}
            for c in new_candidates:
                if c.chunk_id not in existing_ids:
                    obs.candidate_chunks.append(c)
                    existing_ids.add(c.chunk_id)

            # 检查是否有实质进展（新增候选）
            if len(new_candidates) > 0:
                obs.no_progress_count = 0
            else:
                obs.no_progress_count += 1
        else:
            obs.failed_searches.append(search_params.query)
            obs.no_progress_count += 1

        # 更新 subgoal（简化处理）
        obs.current_subgoal = f"Search completed, found {len(new_candidates) if result.success else 0} candidates"
        obs.current_evidence_gap = search_params.target_gap

    def _update_for_read(self, obs: Observation, action: Action, result: ToolResult):
        """READ 动作的状态更新"""
        read_params = action.get_read_params()

        if result.success and result.data:
            summaries = result.data  # List[ReadSummary]

            # 检查是否读到新证据
            existing_ids = {s.chunk_id for s in obs.read_summaries}
            new_evidence_count = 0

            for s in summaries:
                if s.chunk_id not in existing_ids:
                    obs.read_summaries.append(s)
                    existing_ids.add(s.chunk_id)
                    if s.has_new_evidence:
                        new_evidence_count += 1

            # 有效新证据 → no_progress 归零
            if new_evidence_count > 0:
                obs.no_progress_count = 0
            else:
                obs.no_progress_count += 1

        # 检查重复读
        already_read = set(s.chunk_id for s in obs.read_summaries)
        repeated = sum(1 for cid in read_params.chunk_ids if cid in already_read)
        if repeated > 0:
            obs.repeated_action_count += repeated

        obs.current_subgoal = f"Read {len(read_params.chunk_ids)} chunks"

    def _update_for_rerank(self, obs: Observation, action: Action, result: ToolResult):
        """RERANK 动作的状态更新"""
        rerank_params = action.get_rerank_params()

        if result.success and result.data:
            reranked = result.data  # List[CandidateChunk]
            current_ids = [c.chunk_id for c in obs.candidate_chunks]

            # 检查重排是否有实质变化（简化：取前 3 是否相同）
            old_top3 = set(current_ids[:3]) if len(current_ids) >= 3 else set(current_ids)
            new_top3 = set(c.chunk_id for c in reranked[:3])

            if old_top3 == new_top3:
                obs.no_progress_count += 1
            else:
                obs.no_progress_count = 0

            # 更新 candidate_chunks 顺序（保留重排结果）
            reranked_dict = {c.chunk_id: c for c in reranked}
            # 保留只在新候选中的 chunk（reranked 可能不是全部）
            obs.candidate_chunks = [c for cid in current_ids if (c := reranked_dict.get(cid)) is not None]

        obs.current_subgoal = f"Reranked to {len(reranked) if result.success else 0} candidates"

    def _update_for_cite(self, obs: Observation, action: Action, result: ToolResult):
        """CITE 动作的状态更新"""
        cite_params = action.get_cite_params()

        if result.success and result.data:
            cited = result.data["cited_chunk_ids"]
            # 去重追加
            existing = set(obs.cited_chunks)
            for cid in cited:
                if cid not in existing:
                    obs.cited_chunks.append(cid)
                    existing.add(cid)

        obs.current_subgoal = f"Cited {len(obs.cited_chunks)} chunks"

    def _update_for_answer(self, obs: Observation, action: Action, result: ToolResult):
        """ANSWER 动作的状态更新"""
        answer_params = action.get_answer_params()
        obs.final_answer = answer_params.answer_text

        # 合并 cited_chunk_ids
        existing = set(obs.cited_chunks)
        for cid in answer_params.cited_chunk_ids:
            if cid not in existing:
                obs.cited_chunks.append(cid)

        obs.current_subgoal = "Answer generated"


# =============================================================================
# Main Environment
# =============================================================================

class ResearchAgentEnv:
    """
    Research Agent MDP 环境

    持有：
    - corpus_store: 文档语料库
    - tool_registry: 工具注册表
    - reward_computer: reward 计算器
    - current task / observation

    接口：
    - reset(task) -> Observation
    - step(action) -> (Observation, float, bool, dict)
    - finalize_episode() -> reward breakdown
    """

    def __init__(
        self,
        corpus_store: Optional[CorpusStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
        reward_computer: 'RewardComputer' = None,
        max_steps: int = 20
    ):
        self.corpus_store = corpus_store or CorpusStore()
        self.tool_registry = tool_registry or ToolRegistry.create_default(self.corpus_store)
        self.reward_computer = reward_computer
        self.max_steps = max_steps

        self.transition_system = TransitionSystem(self.tool_registry)

        # 当前状态
        self.current_task: Optional[TaskSample] = None
        self.current_obs: Optional[Observation] = None
        self.episode_reward: float = 0.0
        self.step_rewards: List[StepRewardInfo] = []

    def reset(self, task: TaskSample) -> Observation:
        """重置环境，加载新任务"""
        self.current_task = task

        # 初始化 observation
        self.current_obs = Observation(
            task_id=task.task_id,
            task_type=task.task_type,
            user_query=task.user_query,
            remaining_steps=self.max_steps,
            candidate_chunks=[],
            read_summaries=[],
            cited_chunks=[],
            search_history=[],
            failed_searches=[],
            trajectory=[],
        )

        self.episode_reward = 0.0
        self.step_rewards = []

        return self.current_obs

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        执行一步动作

        标准流程：
        1. 校验 action 合法性
        2. 执行工具
        3. 更新 observation
        4. 计算 step reward
        5. 判断 done
        6. 返回 obs, reward, done, info
        """
        info = {}

        # 1. 校验 action 合法性
        is_valid, error_msg = ActionValidator.validate(action, self.current_obs, self.tool_registry)

        if not is_valid:
            # 非法动作处理
            self.current_obs.invalid_action_count += 1

            info["error"] = error_msg
            info["termination_reason"] = TerminationType.INVALID.value if self.current_obs.invalid_action_count >= ActionValidator.MAX_INVALID_ACTIONS else None

            # 更新 step reward
            step_reward = -0.2
            self._record_step_reward(action.tool, step_reward, error_msg)

            # 检查是否达到终止条件
            done = self.current_obs.invalid_action_count >= ActionValidator.MAX_INVALID_ACTIONS

            return self.current_obs, step_reward, done, info

        # 2. 执行工具
        tool = self.tool_registry.get_tool(action.tool)
        tool_result = tool.execute(action.params, context={"observation": self.current_obs})

        # 3. 更新 observation
        self.current_obs = self.transition_system.transition(
            self.current_obs, action, tool_result
        )

        # 4. 计算 step reward
        step_reward = 0.0
        if self.reward_computer:
            step_reward = self.reward_computer.compute_step_reward(
                action=action,
                tool_result=tool_result,
                obs=self.current_obs
            )
        self._record_step_reward(action.tool, step_reward, "step reward")

        self.episode_reward += step_reward

        # 5. 判断 done
        done, termination = self._check_termination()

        info["tool_result"] = tool_result
        info["termination_reason"] = termination.value if termination else None

        # 6. 返回
        return self.current_obs, step_reward, done, info

    def _record_step_reward(self, tool: ToolName, reward: float, reason: str):
        self.step_rewards.append(StepRewardInfo(
            step_idx=len(self.step_rewards),
            tool=tool,
            reward=reward,
            reason=reason
        ))

    def _check_termination(self) -> tuple[bool, Optional[TerminationType]]:
        """
        检查是否满足终止条件
        """
        obs = self.current_obs

        # 1. 执行 ANSWER → 正常终止
        if obs.final_answer is not None:
            return True, TerminationType.ANSWER

        # 2. remaining_steps <= 0
        if obs.remaining_steps <= 0:
            return True, TerminationType.MAX_STEPS

        # 3. invalid_action_count >= 3
        if obs.invalid_action_count >= ActionValidator.MAX_INVALID_ACTIONS:
            return True, TerminationType.INVALID

        # 4. no_progress_count >= 5
        if obs.no_progress_count >= ActionValidator.MAX_NO_PROGRESS:
            return True, TerminationType.NO_PROGRESS

        return False, None

    def finalize_episode(self) -> EpisodeResult:
        """
        Episode 结束后，汇总完整 reward
        """
        if not self.current_task:
            raise RuntimeError("No current task, call reset() first")

        # 计算最终 reward breakdown
        if self.reward_computer:
            reward_breakdown = self.reward_computer.compute_final_reward(
                task=self.current_task,
                obs=self.current_obs
            )
        else:
            reward_breakdown = RewardBreakdown()

        # 构建 EpisodeResult
        trajectory_steps = []
        for record in self.current_obs.trajectory:
            trajectory_steps.append(TrajectoryStep(
                obs=self._obs_to_dict(self.current_obs),
                action={
                    "tool": record.action.tool.value,
                    "intent": record.action.intent,
                    "params": record.action.params,
                    "reasoning": record.action.reasoning
                },
                reward=record.step_reward,
                done=False,
                info={}
            ))

        # 最后一个 step 的 done=True
        if trajectory_steps:
            trajectory_steps[-1].done = True

        episode_result = EpisodeResult(
            task_id=self.current_task.task_id,
            termination=self._check_termination()[1] or TerminationType.MAX_STEPS,
            total_reward=self.episode_reward,
            reward_breakdown=reward_breakdown,
            final_answer=self.current_obs.final_answer,
            cited_chunks=list(self.current_obs.cited_chunks),
            trajectory=trajectory_steps,
        )

        return episode_result

    def _obs_to_dict(self, obs: Observation) -> Dict[str, Any]:
        """Observation -> dict（用于序列化）"""
        return {
            "task_id": obs.task_id,
            "task_type": obs.task_type.value,
            "user_query": obs.user_query,
            "current_subgoal": obs.current_subgoal,
            "current_evidence_gap": obs.current_evidence_gap,
            "remaining_steps": obs.remaining_steps,
            "candidate_chunks": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": c.score, "snippet": c.snippet}
                for c in obs.candidate_chunks
            ],
            "read_summaries": [
                {"chunk_id": s.chunk_id, "doc_id": s.doc_id, "summary": s.summary}
                for s in obs.read_summaries
            ],
            "cited_chunks": list(obs.cited_chunks),
            "search_history": list(obs.search_history),
        }


# =============================================================================
# Reward Computer (占位，后续实现复杂 reward)
# =============================================================================

class RewardComputer:
    """
    Reward 计算器
    MVP 阶段使用简化实现
    """

    def __init__(self, reward_weights: Dict[str, float] = None):
        # 默认权重
        self.weights = reward_weights or {
            "r_final": 1.0,
            "r_citation": 0.3,
            "r_step": 0.1
        }

    def compute_step_reward(self, action: Action, tool_result: ToolResult, obs: Observation) -> float:
        """
        计算单步 reward
        MVP 简化版
        """
        if not tool_result.success:
            return -0.2

        # 根据不同 tool 给不同 reward
        base_rewards = {
            ToolName.SEARCH: 0.01,
            ToolName.READ: 0.05,
            ToolName.RERANK: 0.02,
            ToolName.CITE: 0.03,
            ToolName.ANSWER: 0.0  # ANSWER 的 reward 在 finalize 中计算
        }

        reward = base_rewards.get(action.tool, 0.0)

        # 重复 penalty
        if action.tool == ToolName.SEARCH:
            params = action.get_search_params()
            if params.query in obs.search_history[:-1]:
                reward -= 0.05

        if action.tool == ToolName.READ:
            params = action.get_read_params()
            existing = {s.chunk_id for s in obs.read_summaries}
            repeated = sum(1 for cid in params.chunk_ids if cid in existing)
            if repeated > 0:
                reward -= 0.05 * repeated

        return reward

    def compute_final_reward(
        self,
        task: TaskSample,
        obs: Observation
    ) -> RewardBreakdown:
        """
        计算最终 reward breakdown
        """
        # 1. R_final: 答案质量
        r_final = self._compute_answer_quality(task, obs)

        # 2. R_citation: 引用质量
        r_citation = self._compute_citation_quality(task, obs)

        # 3. R_step_sum: 过程奖励（简化：0）
        r_step_sum = 0.0

        # 总 reward
        r_total = (
            self.weights["r_final"] * r_final +
            self.weights["r_citation"] * r_citation +
            self.weights["r_step"] * r_step_sum
        )

        return RewardBreakdown(
            r_final=r_final,
            r_citation=r_citation,
            r_step_sum=r_step_sum,
            r_total=r_total
        )

    def _compute_answer_quality(self, task: TaskSample, obs: Observation) -> float:
        """
        计算答案质量
        MVP 简化版：基于关键词覆盖和 Rubric 完成度
        """
        if not obs.final_answer:
            return 0.0

        answer_lower = obs.final_answer.lower()
        score = 0.0

        # 关键词覆盖
        if task.rubric:
            keywords = task.rubric.required_keywords
            covered = sum(1 for kw in keywords if kw.lower() in answer_lower)
            keyword_score = covered / len(keywords) if keywords else 0.0
            score += 0.5 * keyword_score

            # section 覆盖
            sections = task.rubric.required_sections
            covered_sections = sum(1 for s in sections if s.lower() in answer_lower)
            section_score = covered_sections / len(sections) if sections else 0.0
            score += 0.3 * section_score

        # 参考答案 overlap
        if task.reference_answer:
            ref_words = set(task.reference_answer.lower().split())
            ans_words = set(answer_lower.split())
            overlap = len(ref_words & ans_words) / max(len(ref_words), 1)
            score += 0.2 * overlap

        return min(1.0, max(0.0, score))

    def _compute_citation_quality(self, task: TaskSample, obs: Observation) -> float:
        """
        计算引用质量
        citation_recall = |cited ∩ gold| / |gold|
        citation_precision = |cited ∩ gold| / |cited|
        """
        cited = set(obs.cited_chunks)
        gold = set(task.gold_chunks)

        if not cited or not gold:
            return 0.0

        intersection = cited & gold
        recall = len(intersection) / len(gold)
        precision = len(intersection) / len(cited)

        return 0.5 * recall + 0.5 * precision
