"""
src/train/rl/reward.py
职责: RL reward 信号计算
设计参考: GAIR-NLP DeepResearcher/tests/e2e/arithmetic_sequence/rl/main_trainer.py
  - make_reward_function 返回 token-level rewards
  - 奖励在 EOS token 处给出（final reward）
  - 支持 dense reward（每步的 evidence_gain 等）

Reward 组成（参考原始设计）:
  - final_answer_reward: answer_quality * 2.0
  - citation_reward: citation_f1 * 1.5
  - evidence_gain_reward: 每步新增 evidence_strength 汇总
  - repeated_action_penalty: -0.1 * n_repeated
  - invalid_action_penalty: -0.2 * n_invalid
  - efficiency_bonus: 0.5 if steps <= optimal_steps
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ...schema.result import EpisodeResult, EvalResult, RewardSignals


@dataclass
class RewardConfig:
    """Reward 权重配置。"""
    final_answer_weight: float = 2.0
    citation_weight: float = 1.5
    evidence_gain_weight: float = 1.0
    repeated_penalty: float = -0.1
    invalid_penalty: float = -0.2
    efficiency_bonus: float = 0.5
    optimal_steps: int = 6
    kl_coef: float = 0.01  # KL penalty coefficient


class RewardFunction:
    """
    Reward 函数接口。

    将 EpisodeResult + EvalResult 转换为 token-level rewards。
    最终奖励在 EOS token 处给出（便于 advantage 计算）。

    使用方式:
        reward_fn = RewardFunction(config=RewardConfig())
        token_rewards = reward_fn.compute_token_rewards(prompt, response)
        # 或者直接用 episode_result
        rewards = reward_fn.compute_from_episode(episode_result, eval_result)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute_from_episode(
        self,
        episode_result: EpisodeResult,
        eval_result: Optional[EvalResult] = None,
    ) -> List[float]:
        """
        从 EpisodeResult 计算 token-level rewards。

        策略:
        - 所有 non-EOS token 的 reward = 0（稀疏 reward）
        - EOS token 的 reward = total_reward（来自 RewardSignals）

        Args:
            episode_result: 环境返回的 episode
            eval_result: 评测结果（可选，不提供时仅用 episode 信息）

        Returns:
            List[float]: token-level rewards，长度为 response_length
        """
        rewards = [0.0] * len(episode_result.trajectory)

        # 计算 RewardSignals
        if eval_result is not None:
            signals = compute_reward_signals(episode_result, eval_result, self.config)
        else:
            signals = self._compute_basic_signals(episode_result)

        total_reward = signals.total

        # 稀疏奖励：在 episode 最后一个有效 step 给予奖励
        # 这里用简单策略：在最后一步给予全部 reward
        if len(rewards) > 0:
            rewards[-1] = total_reward

        return rewards

    def compute_token_rewards(
        self,
        prompt: str,
        response: str,
        episode_result: Optional[EpisodeResult] = None,
        eval_result: Optional[EvalResult] = None,
    ) -> List[float]:
        """
        计算 token-level rewards。

        用于 RolloutBuffer.to_trajectory_samples()。

        简化策略:
        - 如果有 episode_result：在最后给予 sparse reward
        - 否则返回 0
        """
        if episode_result is not None:
            return self.compute_from_episode(episode_result, eval_result)
        return [0.0] * len(response.split())  # 简单占位

    def _compute_basic_signals(
        self,
        episode_result: EpisodeResult,
    ) -> RewardSignals:
        """基于 episode_result 本身计算 reward（无 eval_result 时）。"""
        r = RewardSignals()

        # 基础奖励：基于 done_reason
        if episode_result.done_reason == "answer_submitted":
            r.final_answer_reward = 1.0
        elif episode_result.done_reason == "max_steps":
            r.final_answer_reward = 0.5

        # citation 奖励
        n_cited = len(episode_result.cited_chunk_ids)
        if n_cited > 0:
            r.citation_reward = min(n_cited / 5.0, 1.0) * 1.0

        # evidence gain（从 trajectory 推断）
        evidence_count = 0
        for step in episode_result.trajectory:
            if step.tool_name == "READ" and step.tool_result:
                if hasattr(step.tool_result, "stats"):
                    evidence_count += step.tool_result.stats.get("evidence_gain", 0.0)
        r.evidence_gain_reward = evidence_count

        # penalties
        n_repeated = sum(
            1 for i in range(1, len(episode_result.trajectory))
            if episode_result.trajectory[i].tool_name == episode_result.trajectory[i - 1].tool_name
        )
        r.repeated_action_penalty = self.config.repeated_penalty * n_repeated
        r.invalid_action_penalty = self.config.invalid_penalty * episode_result.n_invalid_steps

        # efficiency bonus
        if episode_result.total_steps <= self.config.optimal_steps:
            r.efficiency_bonus = self.config.efficiency_bonus

        r.total = (
            r.final_answer_reward
            + r.citation_reward
            + r.evidence_gain_reward
            + r.repeated_action_penalty
            + r.invalid_action_penalty
            + r.efficiency_bonus
        )

        return r


def compute_reward_signals(
    episode_result: EpisodeResult,
    eval_result: EvalResult,
    config: Optional[RewardConfig] = None,
) -> RewardSignals:
    """
    计算完整的 RewardSignals。

    公式（参考原始设计）:
        final_answer_reward = answer_quality * 2.0
        citation_reward = citation_f1 * 1.5
        evidence_gain_reward = sum(evidence_strength of new reads) * 1.0
        repeated_action_penalty = -0.1 * n_repeated_actions
        invalid_action_penalty = -0.2 * n_invalid_actions
        efficiency_bonus = 0.5 if total_steps <= optimal_steps

    参考 GAIR-NLP:
    - token_level_scores = reward_tensor
    - batch.batch['token_level_scores'] = reward_tensor (in RayPPOTrainer.fit())
    """
    config = config or RewardConfig()
    r = RewardSignals()

    # final answer reward
    r.final_answer_reward = eval_result.answer_quality * config.final_answer_weight

    # citation reward (F1-based)
    r.citation_reward = eval_result.citation_f1 * config.citation_weight

    # evidence gain (从 trajectory 中 READ 步骤的 evidence_strength 增量)
    evidence_gain = 0.0
    prev_evidence = 0.0
    for step in episode_result.trajectory:
        if step.tool_name == "READ" and step.tool_result:
            if hasattr(step.tool_result, "stats"):
                evidence_gain += step.tool_result.stats.get("evidence_gain", 0.0)
    r.evidence_gain_reward = evidence_gain * config.evidence_gain_weight

    # repeated action penalty
    n_repeated = sum(
        1 for i in range(1, len(episode_result.trajectory))
        if episode_result.trajectory[i].tool_name == episode_result.trajectory[i - 1].tool_name
    )
    r.repeated_action_penalty = config.repeated_penalty * n_repeated

    # invalid action penalty
    r.invalid_action_penalty = config.invalid_penalty * episode_result.n_invalid_steps

    # efficiency bonus
    if episode_result.total_steps <= config.optimal_steps:
        r.efficiency_bonus = config.efficiency_bonus

    # total
    r.total = (
        r.final_answer_reward
        + r.citation_reward
        + r.evidence_gain_reward
        + r.repeated_action_penalty
        + r.invalid_action_penalty
        + r.efficiency_bonus
    )

    return r


class AdaptiveKLController:
    """
    自适应 KL 控制器。

    参考 GAIR-NLP: AdaptiveKLController from https://arxiv.org/pdf/1909.08593.pdf

    作用:
    - 控制策略与参考策略的 KL 散度
    - 当 KL过大时增大系数，过小时减小
    """
    def __init__(
        self,
        init_kl_coef: float = 0.01,
        target_kl: float = 0.1,
        horizon: int = 10000,
    ):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        """
        更新 KL 系数。

        Formula:
            proportional_error = clip(current_kl / target_kl - 1, -0.2, 0.2)
            mult = 1 + proportional_error * n_steps / horizon
            value *= mult
        """
        proportional_error = np.clip(
            current_kl / self.target - 1,
            -0.2,
            0.2,
        )
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

    def compute_kl_penalty(
        self,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray,
    ) -> np.ndarray:
        """
        计算 token-level KL penalty。

        Returns:
            token-level KL divergence (to be subtracted from reward)
        """
        kl = log_probs - ref_log_probs
        return kl * self.value
