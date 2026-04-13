"""
ResearchAgent-RL MVP Run Script
================================
快速跑通 MVP 的入口脚本。

Stage 0: Environment + Mock Tools（验证 obs/action/reward/done/eval）
Stage 1: Rule-based Baseline（生成可执行轨迹）

Usage:
    python scripts/run_mvp.py
"""

import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.schemas import (
    TaskSample, TaskType, Rubric, ToolName, Observation, Action
)
from tools.base import ToolRegistry, CorpusStore, Chunk, Document
from env.environment import ResearchAgentEnv, RewardComputer
from policy.policy import RandomPolicy, RuleBasedPolicy, LLMPolicy
from eval.evaluator import Evaluator


# =============================================================================
# 创建 Mock 数据
# =============================================================================

def create_mock_tasks() -> list[TaskSample]:
    """创建 mock 任务数据"""
    tasks = [
        TaskSample(
            task_id="task_001",
            task_type=TaskType.SURVEY,
            user_query="What are the main approaches to reinforcement learning from human feedback (RLHF)?",
            gold_chunks=["chunk_0", "chunk_1", "chunk_2"],
            rubric=Rubric(
                required_keywords=["RLHF", "reward model", "PPO", "alignment"],
                required_sections=["introduction", "methods", "applications"],
            ),
            reference_answer="RLHF combines RL with human feedback to train language models..."
        ),
        TaskSample(
            task_id="task_002",
            task_type=TaskType.COMPARISON,
            user_query="Compare TRPO and PPO algorithms for policy optimization in RL.",
            gold_chunks=["chunk_3", "chunk_4"],
            rubric=Rubric(
                required_keywords=["TRPO", "PPO", "trust region", "clip"],
                required_sections=["algorithm", "performance", "comparison"],
            ),
            reference_answer="TRPO uses trust region constraint while PPO uses clipped objective..."
        ),
        TaskSample(
            task_id="task_003",
            task_type=TaskType.EXPERIMENT_DESIGN,
            user_query="Design an experiment to evaluate LLM alignment techniques.",
            gold_chunks=["chunk_5", "chunk_6"],
            rubric=Rubric(
                required_keywords=["evaluation", "alignment", "benchmark", "human feedback"],
                required_sections=["setup", "metrics", "baseline"],
            ),
            reference_answer="The experiment should compare alignment techniques using standardized benchmarks..."
        ),
    ]
    return tasks


def create_mock_corpus() -> CorpusStore:
    """创建 mock 语料库"""
    corpus = CorpusStore()

    # 添加一些 mock 文档
    docs = [
        Document(
            doc_id="doc_0",
            title="RLHF: Reinforcement Learning from Human Feedback",
            chunks=[
                Chunk(
                    chunk_id="chunk_0",
                    doc_id="doc_0",
                    content="Reinforcement Learning from Human Feedback (RLHF) is a technique that uses human feedback to train a reward model, which then guides the fine-tuning of a language model. The approach has been popularized by works like InstructGPT and ChatGPT.",
                    title="RLHF: Reinforcement Learning from Human Feedback"
                ),
                Chunk(
                    chunk_id="chunk_1",
                    doc_id="doc_0",
                    content="PPO (Proximal Policy Optimization) is commonly used in RLHF pipelines to optimize the policy using the learned reward model. The clipped surrogate objective prevents destructive large policy updates.",
                    title="RLHF: Reinforcement Learning from Human Feedback"
                ),
            ]
        ),
        Document(
            doc_id="doc_1",
            title="TRPO vs PPO: A Comparison",
            chunks=[
                Chunk(
                    chunk_id="chunk_3",
                    doc_id="doc_1",
                    content="Trust Region Policy Optimization (TRPO) constrains policy updates to a trust region to ensure stability. However, TRPO is computationally expensive due to the conjugate gradient algorithm.",
                    title="TRPO vs PPO: A Comparison"
                ),
                Chunk(
                    chunk_id="chunk_4",
                    doc_id="doc_1",
                    content="Proximal Policy Optimization (PPO) simplifies TRPO by using a clipped objective function. This makes PPO easier to implement and more sample efficient while maintaining similar performance.",
                    title="TRPO vs PPO: A Comparison"
                ),
            ]
        ),
    ]

    for doc in docs:
        corpus.add_document(doc)

    return corpus


# =============================================================================
# 运行单个 episode
# =============================================================================

def run_single_episode(env: ResearchAgentEnv, policy, task: TaskSample, verbose: bool = True):
    """运行单个 episode"""
    obs = env.reset(task)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task.user_query}")
        print(f"Policy: {policy.__class__.__name__}")
        print(f"{'='*60}")

    policy.reset()
    step = 0

    while True:
        step += 1
        action = policy.predict(obs)

        if verbose:
            print(f"\n[Step {step}] {action.tool.value}")
            print(f"  Intent: {action.intent}")
            print(f"  Params: {action.params}")

        obs, reward, done, info = env.step(action)

        if verbose:
            print(f"  Reward: {reward:.3f}")
            print(f"  Done: {done}")
            if info.get("termination_reason"):
                print(f"  Termination: {info['termination_reason']}")

        if done:
            break

    # Finalize episode
    result = env.finalize_episode()

    if verbose:
        print(f"\n--- Episode Result ---")
        print(f"  Termination: {result.termination.value}")
        print(f"  Total Reward: {result.total_reward:.3f}")
        print(f"  R_final: {result.reward_breakdown.r_final:.3f}")
        print(f"  R_citation: {result.reward_breakdown.r_citation:.3f}")
        print(f"  Final Answer: {result.final_answer[:200] if result.final_answer else 'None'}...")
        print(f"  Cited Chunks: {result.cited_chunks}")

    return result


# =============================================================================
# 运行评测
# =============================================================================

def run_evaluation():
    """运行完整评测"""
    print("\n" + "="*60)
    print("Creating mock tasks and corpus...")
    tasks = create_mock_tasks()
    corpus = create_mock_corpus()

    print(f"Created {len(tasks)} tasks, {len(corpus)} chunks in corpus")

    # 创建 environment
    tool_registry = ToolRegistry.create_default(corpus)
    reward_computer = RewardComputer()
    env = ResearchAgentEnv(
        corpus_store=corpus,
        tool_registry=tool_registry,
        reward_computer=reward_computer,
        max_steps=15
    )

    # 评测不同 policy
    policies = {
        "Random": RandomPolicy(tool_registry),
        "RuleBased": RuleBasedPolicy(tool_registry),
    }

    evaluator = Evaluator(tasks)

    for policy_name, policy in policies.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {policy_name}")
        print(f"{'='*60}")

        # 在所有任务上运行
        results = []
        for task in tasks:
            result = run_single_episode(env, policy, task, verbose=False)
            results.append(result)

        # 计算指标
        metrics = evaluator.compute_metrics(results)

        # 生成报告
        report = evaluator.generate_report(metrics)
        print(report)

    # 生成 case study
    print("\n" + "="*60)
    print("Case Study Example:")
    print("="*60)
    task = tasks[0]
    result = run_single_episode(env, RuleBasedPolicy(tool_registry), task, verbose=True)
    case = evaluator.generate_case_study(result)
    print(f"\nTrajectory Summary: {case.trajectory_summary}")
    print(f"Key Decisions: {case.key_decisions}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ResearchAgent-RL MVP Runner")
    print("Stage 0: Environment + Mock Tools")
    print("Stage 1: Rule-based Baseline")
    print()

    run_evaluation()

    print("\n" + "="*60)
    print("MVP smoke test completed!")
    print("="*60)
