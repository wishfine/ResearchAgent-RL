"""
scripts/run_env_demo.py
职责: MVP 端到端演示脚本
设计: 加载 corpus + task；运行 baseline episode；输出评测结果
"""
from __future__ import annotations
import sys
import os

# 添加 src 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.env import ResearchEnv
from src.environment.corpus import CorpusStore
from src.baselines.rule_based import RuleBasedPolicy
from src.tools.search import SearchTool
from src.tools.read import ReadTool
from src.tools.rerank import RerankTool
from src.tools.cite import CiteTool
from src.tools.answer import AnswerTool
from src.eval.evaluator import Evaluator
from src.data.loader import TaskLoader, CorpusLoader
from src.utils.logging import TrajectoryLogger


def create_synthetic_corpus(corpus_dir: str) -> None:
    """创建 synthetic corpus 样例。"""
    import json
    os.makedirs(corpus_dir, exist_ok=True)

    doc1_chunks = [
        {
            "chunk_id": "doc1_c1",
            "doc_id": "doc1",
            "content": "Reinforcement learning from human feedback (RLHF) is a method to train language models. RLHF uses human preferences to guide the learning process.",
            "char_start": 0,
            "char_end": 150,
            "title": "RLHF: Reinforcement Learning from Human Feedback",
            "authors": ["Example Author"],
            "year": 2023,
            "venue": "ACL",
        },
        {
            "chunk_id": "doc1_c2",
            "doc_id": "doc1",
            "content": "Proximal Policy Optimization (PPO) is a popular algorithm for RLHF. PPO balances exploration and exploitation with a clipped objective function.",
            "char_start": 150,
            "char_end": 300,
            "title": "PPO for RLHF",
            "authors": ["Example Author"],
            "year": 2023,
            "venue": "ACL",
        },
    ]

    doc2_chunks = [
        {
            "chunk_id": "doc2_c1",
            "doc_id": "doc2",
            "content": "Deep reinforcement learning has achieved remarkable success in game playing. AlphaGo combines tree search with neural networks.",
            "char_start": 0,
            "char_end": 150,
            "title": "Deep RL in Games",
            "authors": ["Another Author"],
            "year": 2022,
            "venue": "Nature",
        },
        {
            "chunk_id": "doc2_c2",
            "doc_id": "doc2",
            "content": "Reward modeling is critical for RLHF. The reward model predicts human preferences and guides policy optimization.",
            "char_start": 150,
            "char_end": 300,
            "title": "Reward Modeling",
            "authors": ["Another Author"],
            "year": 2022,
            "venue": "Nature",
        },
    ]

    doc3_chunks = [
        {
            "chunk_id": "doc3_c1",
            "doc_id": "doc3",
            "content": "GRPO (Group Relative Policy Optimization) is a new method that simplifies RLHF by removing the need for a critic network. GRPO uses group relative ranking for advantage estimation.",
            "char_start": 0,
            "char_end": 200,
            "title": "GRPO: Group Relative Policy Optimization",
            "authors": ["Third Author"],
            "year": 2024,
            "venue": "ICML",
        },
    ]

    for doc_id, chunks in [("doc1", doc1_chunks), ("doc2", doc2_chunks), ("doc3", doc3_chunks)]:
        doc_data = {
            "doc_id": doc_id,
            "title": chunks[0]["title"].split(":")[0],
            "chunks": chunks,
            "metadata": {},
        }
        with open(os.path.join(corpus_dir, f"{doc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False)


def create_synthetic_task(tasks_dir: str) -> None:
    """创建 synthetic task 样例。"""
    import json
    os.makedirs(tasks_dir, exist_ok=True)

    task_data = {
        "task_id": "synthetic_task_001",
        "task_type": "survey_synthesis",
        "user_query": "What are the main methods for training language models with human feedback?",
        "rubric": {
            "citation_required": True,
            "min_citations": 2,
            "max_citations": 10,
            "require_evidence_for_claims": True,
            "penalize_hallucination": True,
            "task_type_specific": {},
            "answer_weights": {
                "keyword_coverage": 0.4,
                "structural_completeness": 0.3,
                "no_hallucination": 0.3,
            },
        },
        "ground_truth_answer": "RLHF and GRPO are main methods for training language models with human feedback.",
        "ground_truth_citations": ["doc1_c1", "doc2_c2", "doc3_c1"],
        "reference_docs": ["doc1", "doc2", "doc3"],
        "difficulty": "easy",
        "context": "",
        "expected_subgoals": 3,
    }

    with open(os.path.join(tasks_dir, "synthetic_task_001.json"), "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)


def run_demo():
    """运行 MVP 演示。"""
    print("=" * 60)
    print("ResearchAgent-RL MVP Demo")
    print("=" * 60)

    # 路径
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    corpus_dir = os.path.join(base_dir, "data", "corpus")
    tasks_dir = os.path.join(base_dir, "data", "tasks")
    output_dir = os.path.join(base_dir, "outputs")

    # 创建 synthetic 数据
    print("\n[1] Creating synthetic data...")
    create_synthetic_corpus(corpus_dir)
    create_synthetic_task(tasks_dir)
    print(f"    Corpus: {corpus_dir}")
    print(f"    Tasks: {tasks_dir}")

    # 加载数据
    print("\n[2] Loading data...")
    corpus_loader = CorpusLoader()
    corpus = corpus_loader.load(corpus_dir)
    print(f"    Loaded {len(corpus)} chunks")

    task_loader = TaskLoader()
    tasks = task_loader.load_dir(tasks_dir)
    print(f"    Loaded {len(tasks)} tasks")

    if not tasks:
        print("Error: No tasks found")
        return

    task = tasks[0]
    print(f"    Task: {task.task_id} - {task.user_query[:50]}...")

    # 初始化 Environment
    print("\n[3] Initializing Environment...")
    env = ResearchEnv(corpus=corpus, max_steps=15)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(RerankTool())
    env.register_tool(CiteTool())
    env.register_tool(AnswerTool())
    print("    Tools registered: SEARCH, READ, RERANK, CITE, ANSWER")

    # 初始化 Baseline Policy
    policy = RuleBasedPolicy()
    evaluator = Evaluator()
    logger = TrajectoryLogger(output_dir)

    # 运行 Episode
    print("\n[4] Running Episode...")
    obs = env.reset(task)
    print(f"    Initial observation: remaining_steps={obs.remaining_steps}")

    step_count = 0
    for step_count in range(obs.remaining_steps + 1):
        action = policy.decide(obs)
        print(f"    Step {step_count}: {action.tool} - {action.intent[:50]}...")

        obs, done, reason = env.step(action)
        print(f"            done={done}, reason={reason}, remaining={obs.remaining_steps}")

        if done:
            break

    # Finalize Episode
    print("\n[5] Finalizing Episode...")
    episode_result = env.finalize_episode()
    print(f"    Total steps: {episode_result.total_steps}")
    print(f"    Done reason: {episode_result.done_reason}")
    print(f"    Cited chunks: {episode_result.cited_chunk_ids}")

    # Evaluate
    print("\n[6] Evaluating...")
    eval_result = evaluator.evaluate(episode_result, task)
    print(f"    Task success: {eval_result.task_success}")
    print(f"    Answer quality: {eval_result.answer_quality:.3f}")
    print(f"    Citation precision: {eval_result.citation_precision:.3f}")
    print(f"    Citation recall: {eval_result.citation_recall:.3f}")
    print(f"    Citation F1: {eval_result.citation_f1:.3f}")
    print(f"    Search success rate: {eval_result.search_success_rate:.3f}")
    print(f"    Repeated query rate: {eval_result.repeated_query_rate:.3f}")

    # Log
    print("\n[7] Logging results...")
    logger.log_episode(episode_result)
    logger.log_eval(eval_result)
    print(f"    Outputs: {output_dir}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
