#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json

# Add project root to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research_agent.core.env.env import ResearchEnv
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.schema.task import TaskSample, Rubric, TaskType
from research_agent.core.schema.action import Action
from research_agent.core.tools.search import SearchTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.answer import AnswerTool

class SimpleRulePolicy:
    def decide(self, obs) -> Action:
        tid = obs.task_id

        # Scenario 5: parser invalid (toy_task_005)
        # Returns a completely invalid output format on the first step to test parser error detection
        if tid == "toy_task_005" and obs.invalid_action_count == 0:
            return Action(tool="INVALID", intent="Simulate parse error", params={})

        # Scenario 6: max_steps terminate (toy_task_006)
        # Keep searching and never submit an answer to trigger max steps exceeded
        if tid == "toy_task_006":
            return Action.search(query=f"useless query {obs.remaining_steps}", topk=3)

        # 1. If remaining steps is low, answer immediately
        if obs.remaining_steps <= 1:
            return self._make_answer(obs)

        # Scenario 2: SEARCH returns nothing (toy_task_002)
        # We query with a gibberish string on step 0, then submit an answer
        if tid == "toy_task_002":
            if not obs.search_history:
                return Action.search(query="gibberish_query_returning_nothing_123", topk=3)
            elif not obs.candidate_chunks:
                # Search failed, submit final answer
                return Action.answer(answer_text="No info found.", cited_chunk_ids=[])

        # 2. If no candidate chunks, search normally
        if not obs.candidate_chunks:
            return Action.search(query=obs.user_query, topk=3)

        # Scenario 3: READ non-existent chunk_id (toy_task_003)
        # If we have candidates but haven't read yet, try to read a fake chunk ID first
        if tid == "toy_task_003":
            has_read_attempt = "READ" in obs.trajectory
            if not has_read_attempt:
                return Action.read(chunk_ids=["non_existent_chunk_id"])

        # 3. Read unread candidate chunks normally
        read_ids = {s.chunk_id for s in obs.read_summaries}
        unread_chunks = [c.chunk_id for c in obs.candidate_chunks if c.chunk_id not in read_ids]
        
        if unread_chunks:
            return Action.read(chunk_ids=[unread_chunks[0]])

        # 4. If everything read, submit answer
        return self._make_answer(obs)

    def _make_answer(self, obs) -> Action:
        tid = obs.task_id
        
        # Scenario 4: ANSWER with no citations (toy_task_004)
        if tid == "toy_task_004":
            return Action.answer(answer_text="Mars is the answer.", cited_chunk_ids=[])

        cited_chunk_ids = [s.chunk_id for s in obs.read_summaries]
        
        # Construct exact answer if we successfully read the correct chunk
        answers_map = {
            "toy_task_001": "Canada",
            "toy_task_002": "William Shakespeare",
            "toy_task_003": "Paris",
            "toy_task_004": "Mars",
            "toy_task_005": "Albert Einstein",
            "toy_task_006": "H2O",
            "toy_task_007": "Mount Everest",
            "toy_task_008": "Pacific Ocean",
            "toy_task_009": "Leonardo da Vinci",
            "toy_task_010": "1945"
        }
        ans = answers_map.get(tid, "Unknown")
        # Prefix with standard text to check token-level F1 logic
        answer_text = f"The answer is {ans}."
        return Action.answer(answer_text=answer_text, cited_chunk_ids=cited_chunk_ids)


def load_task(filepath: str) -> TaskSample:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TaskSample(
        task_id=data["task_id"],
        task_type=TaskType(data.get("task_type", "survey_synthesis")),
        user_query=data["user_query"],
        rubric=Rubric(),
        ground_truth_answer=data.get("ground_truth_answer"),
        ground_truth_citations=data.get("ground_truth_citations", []),
        reference_docs=data.get("reference_docs", []),
    )


def normalize_answer(s: str) -> str:
    import re
    import string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
        
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_metrics(final_answer: str, ground_truth_answer: str, cited_ids: list[str], gt_citations: list[str]):
    ans_clean = normalize_answer(final_answer) if final_answer else ""
    gt_clean = normalize_answer(ground_truth_answer) if ground_truth_answer else ""
    
    # Strict standardized exact match (EM)
    exact_match = 1.0 if ans_clean == gt_clean else 0.0
    
    # Substring contains match
    contains = 1.0 if gt_clean in ans_clean else 0.0
    
    # Token F1
    token_f1 = compute_token_f1(final_answer, ground_truth_answer)

    # Citation metrics
    cited_set = set(cited_ids)
    gt_set = set(gt_citations)
    
    intersection = cited_set & gt_set
    precision = len(intersection) / len(cited_set) if cited_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    citation_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "contains": contains,
        "token_f1": token_f1,
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": citation_f1
    }


def main():
    print("=" * 60)
    print("ResearchAgent-RL Milestone 0 Smoke Env Run")
    print("=" * 60)

    # 1. Setup paths
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    corpus_dir = os.path.join(base_dir, "data", "searchqa_debug", "corpus")
    tasks_dir = os.path.join(base_dir, "data", "searchqa_debug", "tasks")
    output_dir = os.path.join(base_dir, "outputs", "smoke_test")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load corpus
    print(f"Loading corpus from: {corpus_dir}")
    corpus = CorpusStore(corpus_dir)
    corpus.load()
    print(f"Loaded {len(corpus)} chunks.")

    # 3. Load tasks
    print(f"Loading tasks from: {tasks_dir}")
    task_files = sorted([os.path.join(tasks_dir, f) for f in os.listdir(tasks_dir) if f.endswith(".json")])
    tasks = [load_task(tf) for tf in task_files]
    print(f"Loaded {len(tasks)} tasks.")

    # 4. Initialize Env
    env = ResearchEnv(corpus=corpus, max_steps=10)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(AnswerTool())

    policy = SimpleRulePolicy()

    # 5. Run loop
    results = []
    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/10] Running Task: {task.task_id} - '{task.user_query}'")
        obs = env.reset(task)
        
        step_count = 0
        done = False
        while not done:
            action = policy.decide(obs)
            print(f"  Step {step_count}: Action={action.tool}, Params={action.params}")
            obs, done, reason = env.step(action)
            step_count += 1

        episode_result = env.finalize_episode()
        metrics = compute_metrics(
            episode_result.final_answer,
            task.ground_truth_answer,
            episode_result.cited_chunk_ids,
            task.ground_truth_citations
        )

        print(f"  Done reason: {episode_result.done_reason}")
        print(f"  Metrics: EM={metrics['exact_match']:.2f}, Contains={metrics['contains']:.2f}, Token F1={metrics['token_f1']:.2f}, Citation Prec={metrics['citation_precision']:.2f}, Citation Rec={metrics['citation_recall']:.2f}, Citation F1={metrics['citation_f1']:.2f}")

        # Save trajectory
        task_output_path = os.path.join(output_dir, f"{task.task_id}_trajectory.json")
        with open(task_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": task.to_dict(),
                "episode": episode_result.to_dict(),
                "metrics": metrics
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved trajectory to {task_output_path}")

        results.append((task.task_id, metrics, episode_result))

    # 6. Aggregate results
    avg_em = sum(r[1]["exact_match"] for r in results) / len(results)
    avg_contains = sum(r[1]["contains"] for r in results) / len(results)
    avg_token_f1 = sum(r[1]["token_f1"] for r in results) / len(results)
    avg_cit_f1 = sum(r[1]["citation_f1"] for r in results) / len(results)
    
    print("\n" + "=" * 60)
    print("Smoke test complete! Metric Averages:")
    print(f"  Average Answer EM (Strict): {avg_em:.2f}")
    print(f"  Average Answer Contains: {avg_contains:.2f}")
    print(f"  Average Answer Token F1: {avg_token_f1:.2f}")
    print(f"  Average Citation F1: {avg_cit_f1:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
