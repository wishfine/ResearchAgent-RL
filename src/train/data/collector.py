"""
src/train/data/collector.py
职责: Trajectory 数据收集器
设计: 使用 rule-based policy 或 actor model 收集 trajectories，生成 SFT/RL 训练数据
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
import os
import json
import torch

from ...schema.result import EpisodeResult, EvalResult
from ...schema.task import TaskSample
from ...eval.evaluator import Evaluator
from ...utils.logging import TrajectoryLogger
from .sft.dataset import SFTExample


@dataclass
class CollectedSample:
    """
    收集的单个样本。
    包含 episode_result, eval_result, 可选的 actor_log_probs（如果用模型生成）
    """
    episode_result: EpisodeResult
    eval_result: Optional[EvalResult] = None
    actor_log_probs: Optional[List[float]] = None
    ref_log_probs: Optional[List[float]] = None


class TrajectoryCollector:
    """
    Trajectory 收集器。

    使用环境 + 策略收集 trajectories。
    支持:
    - Rule-based 收集（用于 SFT warmup 或 baseline 数据）
    - Actor model 收集（用于 RL 训练）

    使用方式:
        collector = TrajectoryCollector(
            env_factory=lambda: ResearchEnv(corpus, max_steps=15),
            policy=RuleBasedPolicy(),
            task_sampler=task_sampler,
        )
        samples = collector.collect(n_episodes=100)
    """

    def __init__(
        self,
        env_factory: Callable,  # () -> ResearchEnv
        policy,  # Policy that has decide(obs) -> Action
        task_sampler,  # TaskSampler that has sample() -> TaskSample
        evaluator: Optional[Evaluator] = None,
        max_steps: int = 15,
        save_dir: Optional[str] = None,
    ):
        self.env_factory = env_factory
        self.policy = policy
        self.task_sampler = task_sampler
        self.evaluator = evaluator or Evaluator()
        self.max_steps = max_steps
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def collect(
        self,
        n_episodes: int,
        verbose: bool = False,
    ) -> List[CollectedSample]:
        """
        收集 n_episodes 个样本。

        Args:
            n_episodes: 收集的 episode 数量
            verbose: 是否打印进度

        Returns:
            List[CollectedSample]
        """
        samples = []

        for i in range(n_episodes):
            # 采样 task
            task = self.task_sampler.sample()
            if task is None:
                continue

            # 创建环境
            env = self.env_factory()
            self._register_tools(env)

            # 运行 episode
            episode_result = self._run_episode(env, task)

            # 评测
            eval_result = self.evaluator.evaluate(episode_result, task)

            sample = CollectedSample(
                episode_result=episode_result,
                eval_result=eval_result,
            )
            samples.append(sample)

            # 保存
            if self.save_dir:
                self._save_sample(sample, i)

            if verbose:
                print(f"Episode {i}: done_reason={episode_result.done_reason}, "
                      f"steps={episode_result.total_steps}, "
                      f"citations={len(episode_result.cited_chunk_ids)}, "
                      f"task_success={eval_result.task_success}")

        return samples

    def _register_tools(self, env) -> None:
        """注册工具到环境。"""
        from ...tools.search import SearchTool
        from ...tools.read import ReadTool
        from ...tools.rerank import RerankTool
        from ...tools.cite import CiteTool
        from ...tools.answer import AnswerTool

        env.register_tool(SearchTool())
        env.register_tool(ReadTool())
        env.register_tool(RerankTool())
        env.register_tool(CiteTool())
        env.register_tool(AnswerTool())

    def _run_episode(self, env, task: TaskSample) -> EpisodeResult:
        """运行单个 episode。"""
        obs = env.reset(task)

        for _ in range(self.max_steps):
            action = self.policy.decide(obs)
            obs, done, reason = env.step(action)

            if done:
                break

        return env.finalize_episode()

    def _save_sample(self, sample: CollectedSample, idx: int) -> None:
        """保存单个样本。"""
        filepath = os.path.join(self.save_dir, f"sample_{idx}.json")
        data = {
            "episode_result": sample.episode_result.to_dict(),
            "eval_result": sample.eval_result.to_dict() if sample.eval_result else None,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class ExperienceDataset:
    """
    Experience replay 数据集。

    存储收集的 trajectories，支持:
    - SFT 格式转换
    - RL advantage 重计算
    - Batch 采样
    """

    def __init__(self):
        self.samples: List[CollectedSample] = []

    def add(self, sample: CollectedSample) -> None:
        self.samples.append(sample)

    def add_batch(self, samples: List[CollectedSample]) -> None:
        self.samples.extend(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def to_sft_examples(
        self,
        converter,  # TrajectoryToSFTConverter
    ) -> List[SFTExample]:
        """转换为 SFT 训练样本。"""
        examples = []
        for sample in self.samples:
            example = converter.convert(sample.episode_result)
            examples.append(example)
        return examples

    def filter(
        self,
        min_citations: int = 1,
        min_answer_quality: float = 0.0,
    ) -> "ExperienceDataset":
        """过滤样本。"""
        filtered = ExperienceDataset()
        for sample in self.samples:
            if sample.eval_result is None:
                continue
            if len(sample.episode_result.cited_chunk_ids) < min_citations:
                continue
            if sample.eval_result.answer_quality < min_answer_quality:
                continue
            filtered.add(sample)
        return filtered

    def save(self, filepath: str) -> None:
        """保存到文件。"""
        data = []
        for sample in self.samples:
            data.append({
                "episode_result": sample.episode_result.to_dict(),
                "eval_result": sample.eval_result.to_dict() if sample.eval_result else None,
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "ExperienceDataset":
        """从文件加载。"""
        from ...schema.result import EpisodeResult, EvalResult

        dataset = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            ep_data = item["episode_result"]
            episode_result = EpisodeResult(
                task_id=ep_data["task_id"],
                trajectory=[],  # 不存储 trajectory steps（太大）
                final_answer=ep_data["final_answer"],
                cited_chunk_ids=ep_data["cited_chunk_ids"],
                done_reason=ep_data["done_reason"],
                total_steps=ep_data["total_steps"],
                n_valid_steps=ep_data["n_valid_steps"],
                n_invalid_steps=ep_data["n_invalid_steps"],
                rewards=ep_data.get("rewards", {}),
            )
            eval_result = None
            if item.get("eval_result"):
                er_data = item["eval_result"]
                eval_result = EvalResult(
                    task_id=er_data["task_id"],
                    task_success=er_data["task_success"],
                    answer_quality=er_data["answer_quality"],
                    citation_precision=er_data["citation_precision"],
                    citation_recall=er_data["citation_recall"],
                    citation_f1=er_data["citation_f1"],
                    average_steps=er_data["average_steps"],
                    repeated_query_rate=er_data["repeated_query_rate"],
                    invalid_action_rate=er_data["invalid_action_rate"],
                    search_success_rate=er_data["search_success_rate"],
                    evidence_coverage=er_data["evidence_coverage"],
                    tool_diversity=er_data["tool_diversity"],
                    answer_conciseness=er_data["answer_conciseness"],
                    episode_result=episode_result,
                    rubric_scores=er_data.get("rubric_scores", {}),
                )
            dataset.add(CollectedSample(
                episode_result=episode_result,
                eval_result=eval_result,
            ))
        return dataset
