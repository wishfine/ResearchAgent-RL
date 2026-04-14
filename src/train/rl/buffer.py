"""
src/train/rl/buffer.py
职责: Trajectory 数据缓冲区和样本管理
设计参考: GAIR-NLP DeepResearcher/verl/protocol.py DataProto 设计
  - TrajectorySample 封装单条 RL 样本
  - TrajectoryBuffer 管理批量样本
  - 支持按 episode_id 分组计算 group-level  advantage (GRPO)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict
import torch
import numpy as np


@dataclass
class TrajectorySample:
    """
    单条 RL 训练样本。
    对应一个 query 的一次 rollout（可能包含多个 step）。

    存储:
    - query_ids: tokenized query
    - response_ids: 生成的 response tokens
    - log_probs: response 在当前策略下的 log probabilities
    - advantages: 计算得到的 advantage
    - rewards: 每个 token 的 reward
    - episode_id: 用于 GRPO 分组的 group id
    """
    query_ids: List[int]
    response_ids: List[int]
    input_ids: List[int] = field(default_factory=list)  # query + response
    attention_mask: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)  # 每个 token 的 log_prob
    ref_log_probs: List[float] = field(default_factory=list)  # reference model log_prob
    rewards: List[float] = field(default_factory=list)  # 每个 token 的 reward
    advantages: List[float] = field(default_factory=list)  # 每个 token 的 advantage
    returns: List[float] = field(default_factory=list)  # 每个 token 的 return
    episode_id: str = ""  # 用于 GRPO 分组
    task_id: str = ""
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tensors(self, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """转换为 PyTorch tensors。"""
        return {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "ref_log_probs": torch.tensor(self.ref_log_probs, dtype=torch.float32, device=device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(self.returns, dtype=torch.float32, device=device),
        }

    @property
    def response_length(self) -> int:
        return len(self.response_ids)

    @property
    def is_valid(self) -> bool:
        """检查样本是否有效（有 response）。"""
        return len(self.response_ids) > 0


@dataclass
class TrajectoryBuffer:
    """
    Trajectory 缓冲区。

    管理一个 batch 的 TrajectorySample。
    支持:
    - 添加样本
    - 按 episode_id 分组（GRPO）
    - 计算 group-level advantages
    - 转换为 tensor batch

    参考 GAIR-NLP:
    - DataProto.batch 作为 TensorDict
    - DataProto.non_tensor_batch 存储元数据
    - 支持 repeat() 进行 GRPO 风格的 group 扩展
    """

    def __init__(self):
        self.samples: List[TrajectorySample] = []
        self._grouped: Dict[str, List[TrajectorySample]] = defaultdict(list)

    def add(self, sample: TrajectorySample) -> None:
        """添加一个样本。"""
        self.samples.append(sample)
        self._grouped[sample.episode_id].append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def get_group(self, episode_id: str) -> List[TrajectorySample]:
        """获取同一 episode 的所有样本（用于 GRPO）。"""
        return self._grouped.get(episode_id, [])

    @property
    def episode_ids(self) -> List[str]:
        """获取所有 episode id。"""
        return list(self._grouped.keys())

    def compute_group_advantages(
        self,
        epsilon: float = 1e-6,
        advantage_normalize: bool = True,
    ) -> None:
        """
        计算 group-level advantages（GRPO 风格）。

        对同一 episode_id 的样本，计算相对优势：
        advantage_i = (reward_i - group_mean) / (group_std + epsilon)

        参考 GAIR-NLP compute_grpo_outcome_advantage():
        - 对同一 prompt 的多次生成做归一化
        - 使用 reward sum 作为 group statistic
        """
        for episode_id, group_samples in self._grouped.items():
            if len(group_samples) < 2:
                # 单样本无法计算相对优势
                continue

            # 计算每条样本的 total reward（scalar）
            total_rewards = []
            for sample in group_samples:
                # 使用 episode 总 reward（最后一个 token 的 cumsum）
                if len(sample.rewards) > 0:
                    total_reward = sum(sample.rewards)
                else:
                    total_reward = 0.0
                total_rewards.append(total_reward)

            total_rewards = np.array(total_rewards)
            mean_reward = total_rewards.mean()
            std_reward = total_rewards.std() + epsilon

            # 计算相对优势
            for i, sample in enumerate(group_samples):
                normalized_adv = (total_rewards[i] - mean_reward) / std_reward

                # 将标量 advantage 扩展到 token 级别
                if len(sample.response_ids) > 0:
                    sample.advantages = [normalized_adv] * len(sample.response_ids)

    def to_tensor_batch(self, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """
        将 buffer 转换为 tensor batch。

        Returns:
        {
            "input_ids": [batch_size, seq_len]
            "attention_mask": [batch_size, seq_len]
            "log_probs": [batch_size, seq_len]
            "ref_log_probs": [batch_size, seq_len]
            "rewards": [batch_size, seq_len]
            "advantages": [batch_size, seq_len]
            "returns": [batch_size, seq_len]
        }
        """
        if len(self.samples) == 0:
            return {}

        # Pad 到相同长度
        max_len = max(len(s.input_ids) for s in self.samples)
        batch_size = len(self.samples)

        # 初始化 tensors
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)
        ref_log_probs = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)
        rewards = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)
        advantages = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)
        returns = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)

        for i, sample in enumerate(self.samples):
            seq_len = len(sample.input_ids)
            input_ids[i, :seq_len] = torch.tensor(sample.input_ids, device=device)
            attention_mask[i, :seq_len] = torch.tensor(sample.attention_mask, device=device)
            log_probs[i, :seq_len] = torch.tensor(sample.log_probs, device=device)
            ref_log_probs[i, :seq_len] = torch.tensor(sample.ref_log_probs, device=device)
            rewards[i, :seq_len] = torch.tensor(sample.rewards, device=device)
            advantages[i, :seq_len] = torch.tensor(sample.advantages, device=device)
            returns[i, :seq_len] = torch.tensor(sample.returns, device=device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
            "rewards": rewards,
            "advantages": advantages,
            "returns": returns,
        }

    def repeat(self, repeat_times: int = 2, interleave: bool = True) -> "TrajectoryBuffer":
        """
        扩展 buffer 进行 GRPO-style 训练。

        对每个样本重复 n 次，保持 episode_id 相同以便计算 group advantage。

        Args:
            repeat_times: 每个样本重复次数
            interleave: 是否交替排列（保持 group 内样本分散）

        Returns:
            新的扩展后的 TrajectoryBuffer
        """
        new_buffer = TrajectoryBuffer()

        if interleave:
            # 交替排列：sample0_rep0, sample1_rep0, ..., sample0_rep1, sample1_rep1, ...
            for rep in range(repeat_times):
                for sample in self.samples:
                    new_sample = TrajectorySample(
                        query_ids=sample.query_ids.copy(),
                        response_ids=sample.response_ids.copy(),
                        input_ids=sample.input_ids.copy(),
                        attention_mask=sample.attention_mask.copy(),
                        log_probs=sample.log_probs.copy(),
                        ref_log_probs=sample.ref_log_probs.copy(),
                        rewards=sample.rewards.copy(),
                        advantages=sample.advantages.copy(),
                        returns=sample.returns.copy(),
                        episode_id=sample.episode_id,  # 保持相同 episode_id
                        task_id=sample.task_id,
                        metadata=sample.metadata.copy(),
                    )
                    new_buffer.add(new_sample)
        else:
            # 按顺序排列：sample0 所有重复, sample1 所有重复, ...
            for sample in self.samples:
                for _ in range(repeat_times):
                    new_sample = TrajectorySample(
                        query_ids=sample.query_ids.copy(),
                        response_ids=sample.response_ids.copy(),
                        input_ids=sample.input_ids.copy(),
                        attention_mask=sample.attention_mask.copy(),
                        log_probs=sample.log_probs.copy(),
                        ref_log_probs=sample.ref_log_probs.copy(),
                        rewards=sample.rewards.copy(),
                        advantages=sample.advantages.copy(),
                        returns=sample.returns.copy(),
                        episode_id=sample.episode_id,
                        task_id=sample.task_id,
                        metadata=sample.metadata.copy(),
                    )
                    new_buffer.add(new_sample)

        return new_buffer


class RolloutBuffer:
    """
    Rollout 阶段的数据缓冲。

    在 RL 循环中，用于收集 actor 生成的样本。
    每个样本包含:
    - prompt (task)
    - generated response
    - log_probs under current policy
    - log_probs under reference policy
    """

    def __init__(self):
        self.prompts: List[str] = []
        self.responses: List[str] = []
        self.response_ids: List[List[int]] = []
        self.log_probs: List[List[float]] = []
        self.ref_log_probs: List[List[float]] = []
        self.episode_ids: List[str] = []
        self.task_ids: List[str] = []

    def add(
        self,
        prompt: str,
        response: str,
        response_ids: List[int],
        log_probs: List[float],
        ref_log_probs: List[float],
        episode_id: str,
        task_id: str,
    ) -> None:
        """添加一个 rollout 样本。"""
        self.prompts.append(prompt)
        self.responses.append(response)
        self.response_ids.append(response_ids)
        self.log_probs.append(log_probs)
        self.ref_log_probs.append(ref_log_probs)
        self.episode_ids.append(episode_id)
        self.task_ids.append(task_id)

    def __len__(self) -> int:
        return len(self.prompts)

    def to_trajectory_samples(
        self,
        tokenizer,
        reward_fn,
        max_length: int = 2048,
    ) -> List[TrajectorySample]:
        """
        将 rollout 数据转换为 TrajectorySamples，并计算 reward。

        Args:
            tokenizer: 用于 tokenize
            reward_fn: RewardFunction 实例，计算 token-level rewards

        Returns:
            List[TrajectorySample] with rewards computed
        """
        samples = []

        for i in range(len(self)):
            prompt = self.prompts[i]
            response = self.responses[i]
            response_ids = self.response_ids[i]
            lp = self.log_probs[i]
            ref_lp = self.ref_log_probs[i]
            episode_id = self.episode_ids[i]
            task_id = self.task_ids[i]

            # Tokenize prompt
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            full_ids = prompt_ids + response_ids
            attention_mask = [1] * len(full_ids)

            # Pad to max_length
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                lp = lp[:max_length - len(prompt_ids)]
                ref_lp = ref_lp[:max_length - len(prompt_ids)]
            else:
                pad_len = max_length - len(full_ids)
                full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                lp = lp + [0.0] * pad_len
                ref_lp = ref_lp + [0.0] * pad_len

            # 计算 rewards
            rewards = reward_fn.compute_token_rewards(prompt, response)

            # Pad rewards
            if len(rewards) > max_length:
                rewards = rewards[:max_length]
            else:
                rewards = rewards + [0.0] * (max_length - len(rewards))

            sample = TrajectorySample(
                query_ids=prompt_ids,
                response_ids=response_ids,
                input_ids=full_ids,
                attention_mask=attention_mask,
                log_probs=lp,
                ref_log_probs=ref_lp,
                rewards=rewards[:max_length],
                advantages=[0.0] * max_length,  # 待计算
                returns=[0.0] * max_length,  # 待计算
                episode_id=episode_id,
                task_id=task_id,
            )
            samples.append(sample)

        return samples
