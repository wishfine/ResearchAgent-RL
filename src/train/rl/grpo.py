"""
src/train/rl/grpo.py
职责: GRPO (Group Relative Policy Optimization) 训练器
设计参考: GAIR-NLP DeepResearcher/verl/trainer/ppo/ray_trainer.py
  - 使用 group-level advantage computation
  - 无需 critic (value function)
  - PPO-style policy update with clip
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .buffer import TrajectoryBuffer, TrajectorySample, RolloutBuffer
from .reward import RewardFunction, RewardConfig, AdaptiveKLController, compute_reward_signals


@dataclass
class GRPOConfig:
    """GRPO 训练配置。"""
    # 训练
    output_dir: str = "./outputs/rl"
    num_epochs: int = 10
    num_episodes_per_epoch: int = 100
    num_envs_perEpisode: int = 4  # 每个 task 生成的样本数 (GRPO group size)
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_steps: int = -1

    # RL 算法
    gamma: float = 1.0  # discount factor (GRPO 通常 = 1.0)
    lam: float = 0.95  # GAE lambda (GRPO 不使用，但保留接口)
    clip_ratio: float = 0.2  # PPO clip ratio
    entropy_coef: float = 0.01  # entropy bonus
    kl_coef: float = 0.01  # KL penalty coefficient
    target_kl: float = 0.1  # 目标 KL (用于 early stopping)
    max_kl_div: float = 0.15  # early stopping threshold

    # 生成
    max_response_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True

    # 其他
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = False
    bf16: bool = True
    save_total_limit: int = 2


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) 训练器。

    参考 GAIR-NLP RayPPOTrainer.fit() 中的 GRPO 流程:

    1. Rollout: 对每个 task 生成 num_envs_per_episode 个样本
       - 使用当前 actor 模型生成 response
       - 计算 log_probs (actor) 和 ref_log_probs (reference)

    2. Reward: 使用 RewardFunction 计算 token-level rewards

    3. Advantage Computation:
       - 对同一 episode_id (task) 的样本，计算 group-level advantage
       - advantage_i = (reward_i - group_mean) / (group_std + eps)

    4. Policy Update:
       - 对扩展后的 batch（每样本重复 n 次）进行 PPO-style update
       - loss = -min(ratio * advantage, clamp(ratio, 1-clip, 1+clip) * advantage)
       - + entropy bonus
       - + KL penalty

    使用方式:
        trainer = GRPOTrainer(
            actor_model=actor_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            env_factory=lambda: ResearchEnv(corpus, max_steps=15),
            config=GRPOConfig(),
        )
        trainer.train()
    """

    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        env_factory: Callable,  # () -> ResearchEnv，不实例化以便多次创建
        reward_fn: Optional[RewardFunction] = None,
        config: Optional[GRPOConfig] = None,
    ):
        self.actor_model = actor_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.env_factory = env_factory
        self.reward_fn = reward_fn or RewardFunction()
        self.config = config or GRPOConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_model.to(self.device)
        self.ref_model.to(self.device)
        self.ref_model.eval()  # reference model 不更新

        self.kl_controller = AdaptiveKLController(
            init_kl_coef=self.config.kl_coef,
            target_kl=self.config.target_kl,
        )

        self.optimizer = torch.optim.AdamW(
            self.actor_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        os.makedirs(self.config.output_dir, exist_ok=True)

    def train(self) -> None:
        """主训练循环。"""
        global_step = 0
        total_loss = 0.0
        total_reward = 0.0
        pbar = tqdm(total=self.config.max_steps if self.config.max_steps > 0 else None)

        for epoch in range(self.config.num_epochs):
            for episode_idx in range(self.config.num_episodes_per_epoch):
                # 1. Rollout: 生成一个 batch 的样本
                rollout_buffer = self._rollout(episode_idx)

                if len(rollout_buffer) == 0:
                    continue

                # 2. 转换为 TrajectorySamples 并计算 reward
                trajectory_buffer = self._compute_rewards(rollout_buffer)

                # 3. 计算 group-level advantages
                trajectory_buffer.compute_group_advantages()

                # 4. PPO-style policy update
                update_stats = self._update_policy(trajectory_buffer)

                # 统计
                global_step += 1
                total_loss += update_stats["loss"]
                total_reward += update_stats["mean_reward"]

                if global_step % self.config.logging_steps == 0:
                    pbar.set_postfix({
                        "loss": f"{total_loss / self.config.logging_steps:.4f}",
                        "reward": f"{total_reward / self.config.logging_steps:.4f}",
                        "kl": f"{update_stats.get('mean_kl', 0):.4f}",
                    })
                    total_loss = 0.0
                    total_reward = 0.0

                if global_step % self.config.save_steps == 0:
                    self.save(f"{self.config.output_dir}/checkpoint-{global_step}")

                if self.config.max_steps > 0 and global_step >= self.config.max_steps:
                    break

                pbar.update(1)

        pbar.close()
        self.save(f"{self.config.output_dir}/final")

    def _rollout(self, episode_idx: int) -> RolloutBuffer:
        """
        Rollout 阶段：使用当前策略生成样本。

        对每个 task（episode_id）生成 num_envs_per_episode 个样本。

        Returns:
            RolloutBuffer 包含生成的样本和 log_probs
        """
        buffer = RolloutBuffer()
        self.actor_model.eval()
        self.ref_model.eval()

        # 获取 task（这里简化处理，实际应从 task dataset 获取）
        # TODO: 需要 task_sampler 接口
        task_sample = self._sample_task(episode_idx)
        if task_sample is None:
            return buffer

        with torch.no_grad():
            for env_idx in range(self.config.num_episodes_per_episode):
                # 为每个样本创建环境实例
                env = self.env_factory()
                env.reset(task_sample)

                episode_id = f"epoch_{episode_idx}_env_{env_idx}_task_{task_sample.task_id}"
                task_id = task_sample.task_id

                # 使用 actor 模型生成完整 episode
                episode_result, rollout_tokens, log_probs = self._generate_episode(
                    env, task_sample, episode_id, task_id
                )

                # 计算 reference log probs
                ref_log_probs = self._compute_ref_log_probs(rollout_tokens)

                buffer.add(
                    prompt=task_sample.user_query,
                    response=episode_result.final_answer,
                    response_ids=rollout_tokens,
                    log_probs=log_probs,
                    ref_log_probs=ref_log_probs,
                    episode_id=episode_id,
                    task_id=task_id,
                )

        return buffer

    def _generate_episode(
        self,
        env,
        task_sample,
        episode_id: str,
        task_id: str,
    ):
        """
        使用 actor 模型生成一个完整 episode。

        简化实现：
        - 使用 actor model 生成 action
        - 执行 action 获取 observation
        - 重复直到 done

        返回:
            episode_result, response_ids, log_probs
        """
        obs = env.reset(task_sample)
        response_ids = []
        log_probs_list = []

        # 简化：用 rule-based policy 作为基线
        # 实际应该用 actor model 生成
        # TODO: 接入 LLM actor
        from ...baselines.rule_based import RuleBasedPolicy
        policy = RuleBasedPolicy()

        max_tokens = self.config.max_response_length

        for step in range(env.max_steps):
            # 使用 policy 决定 action（简化版，暂不用 LLM actor）
            action = policy.decide(obs)
            action_str = f"{action.tool}: {action.intent[:50]}"

            # Tokenize action
            action_tokens = self.tokenizer.encode(
                action_str,
                add_special_tokens=False,
            )

            # 截断
            if len(response_ids) + len(action_tokens) > max_tokens:
                break

            response_ids.extend(action_tokens)
            # 简化：假设每个 token 的 log_prob = -0.1
            log_probs_list.extend([-0.1] * len(action_tokens))

            # 执行 action
            obs, done, reason = env.step(action)

            if done:
                break

        # Finalize
        episode_result = env.finalize_episode()

        return episode_result, response_ids, log_probs_list

    def _compute_ref_log_probs(self, token_ids: List[int]) -> List[float]:
        """计算 reference model 的 log probs。"""
        if len(token_ids) == 0:
            return []

        input_ids = torch.tensor([token_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            # 计算 log probs
            log_probs = torch.log_softmax(logits, dim=-1)

            # 获取每个 token 的 log prob
            ref_log_probs = []
            for i in range(len(token_ids)):
                if i < log_probs.shape[0]:
                    ref_log_probs.append(log_probs[i, token_ids[i]].item())
                else:
                    ref_log_probs.append(0.0)

        return ref_log_probs

    def _compute_rewards(self, rollout_buffer: RolloutBuffer) -> TrajectoryBuffer:
        """将 rollout 数据转换为 TrajectorySamples 并计算 rewards。"""
        trajectory_buffer = TrajectoryBuffer()

        for i in range(len(rollout_buffer)):
            sample = TrajectorySample(
                query_ids=self.tokenizer.encode(rollout_buffer.prompts[i], add_special_tokens=True),
                response_ids=rollout_buffer.response_ids[i],
                input_ids=rollout_buffer.response_ids[i],  # 简化
                attention_mask=[1] * len(rollout_buffer.response_ids[i]),
                log_probs=rollout_buffer.log_probs[i],
                ref_log_probs=rollout_buffer.ref_log_probs[i],
                rewards=[0.0] * len(rollout_buffer.response_ids[i]),  # 待填充
                episode_id=rollout_buffer.episode_ids[i],
                task_id=rollout_buffer.task_ids[i],
            )

            # 计算 reward（在最后一个 token）
            if len(sample.response_ids) > 0:
                rewards = [0.0] * (len(sample.response_ids) - 1)
                rewards.append(1.0)  # placeholder reward
                sample.rewards = rewards

            trajectory_buffer.add(sample)

        return trajectory_buffer

    def _update_policy(self, trajectory_buffer: TrajectoryBuffer) -> Dict[str, float]:
        """
        PPO-style policy update。

        步骤:
        1. repeat buffer (GRPO-style)
        2. 计算 advantage
        3. 计算 policy loss (PPO clip loss)
        4. 计算 entropy loss
        5. 计算 KL penalty
        6. backward and step
        """
        self.actor_model.train()

        # 扩展 buffer（每个样本重复多次，保持 episode_id 相同）
        expanded_buffer = trajectory_buffer.repeat(
            repeat_times=self.config.num_episodes_per_episode,
            interleave=True,
        )

        # 转换为 tensors
        batch = expanded_buffer.to_tensor_batch(self.device)
        advantages = batch["advantages"]
        old_log_probs = batch["log_probs"]

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # PPO update
        batch_size = batch["input_ids"].shape[0]
        mini_batch_size = self.config.per_device_train_batch_size
        num_updates = 0
        total_loss = 0.0
        total_kl = 0.0

        for epoch in range(self.config.num_epochs):
            # Shuffle
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                # Get mini-batch
                mb_input_ids = batch["input_ids"][mb_indices]
                mb_attention_mask = batch["attention_mask"][mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Forward pass
                outputs = self.actor_model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                )
                logits = outputs.logits  # [batch, seq_len, vocab_size]

                # 计算 log_probs
                log_probs = torch.log_softmax(logits, dim=-1)

                # 获取每个 token 的 log_prob（与 old_log_probs 相同位置）
                # 简化：取 response 部分的 log_prob
                seq_len = mb_input_ids.shape[1]
                mb_new_log_probs = log_probs.view(-1, log_probs.size(-1))
                mb_old_flat = mb_old_log_probs.view(-1)

                # 计算 ratio
                ratio = torch.exp(mb_new_log_probs - mb_old_flat.unsqueeze(-1))

                # PPO clip loss
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_ratio,
                    1 + self.config.clip_ratio,
                )
                pg_loss = -torch.min(
                    ratio * mb_advantages.view(-1, 1),
                    clipped_ratio * mb_advantages.view(-1, 1),
                ).mean()

                # Entropy bonus
                entropy = -(torch.log_softmax(logits, dim=-1) * torch.softmax(logits, dim=-1)).sum(dim=-1).mean()

                # KL penalty
                kl_div = (torch.exp(mb_old_log_probs) * (mb_old_log_probs - log_probs)).sum(dim=-1).mean()

                # Total loss
                loss = pg_loss - self.config.entropy_coef * entropy + self.config.kl_coef * kl_div

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_kl += kl_div.item()
                num_updates += 1

                # Early stopping based on KL
                if kl_div > self.config.max_kl_div:
                    break

        return {
            "loss": total_loss / max(num_updates, 1),
            "mean_kl": total_kl / max(num_updates, 1),
            "mean_reward": 0.0,  # placeholder
        }

    def _sample_task(self, idx: int):
        """
        采样一个 task。

        TODO: 接入 TaskDataset
        简化返回 None，实际需要从 task dataset 获取
        """
        # 暂时返回 None，实际使用时需要实现 task sampler
        return None

    def save(self, output_dir: str) -> None:
        """保存 actor 模型。"""
        os.makedirs(output_dir, exist_ok=True)
        self.actor_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, checkpoint_dir: str) -> None:
        """加载 checkpoint。"""
        from transformers import AutoModelForCausalLM
        self.actor_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        self.actor_model.to(self.device)
