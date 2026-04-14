"""
scripts/run_training.py
职责: 端到端 SFT 和 RL 训练脚本
使用方式:
    # SFT
    python scripts/run_training.py --mode sft --model_name_or_path <model> --train_file <data.jsonl>

    # RL (GRPO)
    python scripts/run_training.py --mode rl --model_name_or_path <model>
"""
from __future__ import annotations
import sys
import os
import argparse
from dataclasses import dataclass

# 添加 src 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_sft(args):
    """运行 SFT 训练。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.train.sft.trainer import SFTTrainer, SFTConfig
    from src.train.data.collector import ExperienceDataset
    from src.train.sft.dataset import TrajectoryToSFTConverter, SFTDatasetWriter

    print("=" * 60)
    print("SFT Training")
    print("=" * 60)

    # 加载模型
    print(f"\n[1] Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备 SFT 数据
    print(f"\n[2] Preparing SFT data from: {args.train_file}")

    # 如果需要从 collected data 转换
    if args.train_file.endswith(".json") and "collected" in args.train_file:
        print("    Converting collected data to SFT format...")
        dataset = ExperienceDataset.load(args.train_file)
        converter = TrajectoryToSFTConverter()
        examples = dataset.to_sft_examples(converter)
        writer = SFTDatasetWriter(os.path.dirname(args.train_file))
        sft_file = writer.write(examples, "sft_data.jsonl")
        print(f"    SFT data written to: {sft_file}")
    else:
        sft_file = args.train_file

    # 配置
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        bf16=args.bf16,
    )

    # 训练
    print(f"\n[3] Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_files=sft_file,
        config=config,
    )
    trainer.train()
    trainer.save()

    print(f"\n[4] SFT training complete! Model saved to: {args.output_dir}")


def run_rl(args):
    """运行 RL (GRPO) 训练。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    from src.environment.corpus import CorpusStore
    from src.data.loader import TaskLoader, CorpusLoader
    from src.train.rl.grpo import GRPOTrainer, GRPOConfig
    from src.train.rl.reward import RewardFunction, RewardConfig

    print("=" * 60)
    print("RL (GRPO) Training")
    print("=" * 60)

    # 加载模型
    print(f"\n[1] Loading models: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    ref_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_model.to(device)

    # 加载 corpus
    print(f"\n[2] Loading corpus from: {args.corpus_dir}")
    corpus_loader = CorpusLoader()
    corpus = corpus_loader.load(args.corpus_dir)
    print(f"    Loaded {len(corpus)} chunks")

    # 创建环境工厂
    from src.environment.env import ResearchEnv

    def env_factory():
        return ResearchEnv(corpus=corpus, max_steps=15)

    # 加载 task sampler
    print(f"\n[3] Loading tasks from: {args.tasks_dir}")
    task_loader = TaskLoader()
    tasks = task_loader.load_dir(args.tasks_dir)
    print(f"    Loaded {len(tasks)} tasks")

    # Task sampler
    import random

    class RandomTaskSampler:
        def __init__(self, tasks):
            self.tasks = tasks

        def sample(self):
            return random.choice(self.tasks)

    task_sampler = RandomTaskSampler(tasks)

    # Reward function
    reward_fn = RewardFunction(config=RewardConfig())

    # GRPO 配置
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        num_episodes_per_epoch=args.num_episodes_per_epoch,
        num_envs_per_episode=args.num_envs_per_episode,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
    )

    # 训练
    print(f"\n[4] Starting GRPO training...")
    trainer = GRPOTrainer(
        actor_model=actor_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        env_factory=env_factory,
        reward_fn=reward_fn,
        config=config,
    )
    trainer.train()

    print(f"\n[5] RL training complete! Model saved to: {args.output_dir}")


def collect_data(args):
    """收集训练数据。"""
    from src.environment.corpus import CorpusStore
    from src.data.loader import TaskLoader, CorpusLoader
    from src.baselines.rule_based import RuleBasedPolicy
    from src.train.data.collector import TrajectoryCollector, ExperienceDataset

    print("=" * 60)
    print("Data Collection (Rule-Based)")
    print("=" * 60)

    # 加载 corpus
    print(f"\n[1] Loading corpus from: {args.corpus_dir}")
    corpus_loader = CorpusLoader()
    corpus = corpus_loader.load(args.corpus_dir)
    print(f"    Loaded {len(corpus)} chunks")

    # 环境工厂
    from src.environment.env import ResearchEnv

    def env_factory():
        return ResearchEnv(corpus=corpus, max_steps=15)

    # 加载 tasks
    print(f"\n[2] Loading tasks from: {args.tasks_dir}")
    task_loader = TaskLoader()
    tasks = task_loader.load_dir(args.tasks_dir)
    print(f"    Loaded {len(tasks)} tasks")

    # Task sampler
    import random

    class RandomTaskSampler:
        def __init__(self, tasks):
            self.tasks = tasks

        def sample(self):
            return random.choice(self.tasks)

    task_sampler = RandomTaskSampler(tasks)

    # 收集器
    policy = RuleBasedPolicy()
    collector = TrajectoryCollector(
        env_factory=env_factory,
        policy=policy,
        task_sampler=task_sampler,
        max_steps=15,
        save_dir=args.output_dir,
    )

    # 收集数据
    print(f"\n[3] Collecting {args.n_episodes} episodes...")
    samples = collector.collect(n_episodes=args.n_episodes, verbose=True)

    # 保存
    dataset = ExperienceDataset()
    for sample in samples:
        dataset.add(sample)

    output_file = os.path.join(args.output_dir, "collected_data.json")
    dataset.save(output_file)
    print(f"\n[4] Data collection complete! Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="ResearchAgent-RL Training")
    parser.add_argument("--mode", type=str, choices=["sft", "rl", "collect"],
                        default="collect", help="Training mode")
    parser.add_argument("--model_name_or_path", type=str,
                        default="gpt2", help="Model name or path")
    parser.add_argument("--corpus_dir", type=str,
                        default="data/corpus", help="Corpus directory")
    parser.add_argument("--tasks_dir", type=str,
                        default="data/tasks", help="Tasks directory")
    parser.add_argument("--train_file", type=str,
                        default="outputs/sft/sft_data.jsonl", help="SFT train file")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs", help="Output directory")
    parser.add_argument("--n_episodes", type=int,
                        default=100, help="Number of episodes to collect")

    # SFT args
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=True)

    # RL args
    parser.add_argument("--num_episodes_per_epoch", type=int, default=100)
    parser.add_argument("--num_envs_per_episode", type=int, default=4)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=-1)

    args = parser.parse_args()

    if args.mode == "sft":
        run_sft(args)
    elif args.mode == "rl":
        run_rl(args)
    elif args.mode == "collect":
        collect_data(args)


if __name__ == "__main__":
    main()
