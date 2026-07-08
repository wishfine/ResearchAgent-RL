#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import json
import argparse

# Add project root to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research_agent.core.env.env import ResearchEnv
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.schema.task import TaskSample, Rubric, TaskType
from research_agent.core.schema.parser import ActionParser
from research_agent.core.env.llm_client import LLMClient
from research_agent.core.env.rollout_collector import ConversationCollector
from research_agent.core.env.mask_builder import build_loss_mask, MockTokenizer
from research_agent.core.tools.search import SearchTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.answer import AnswerTool

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
    
    exact_match = 1.0 if ans_clean == gt_clean else 0.0
    contains = 1.0 if gt_clean in ans_clean else 0.0
    token_f1 = compute_token_f1(final_answer, ground_truth_answer)

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

def visualize_mask(input_ids: list[int], loss_mask: list[int], max_len: int = 150):
    print("\nToken Loss Mask Visualization (first 150 chars):")
    print("-" * 80)
    for idx in range(min(len(input_ids), max_len)):
        char = chr(input_ids[idx])
        mask = loss_mask[idx]
        char_rep = repr(char)[1:-1]
        if char == "\n":
            char_rep = "\\n"
        if mask == 1:
            print(f"\033[92m{char_rep}\033[0m", end="")
        else:
            print(f"\033[90m{char_rep}\033[0m", end="")
    print("\n" + "-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Qwen Zero-shot Rollout Client")
    parser.add_argument("--model_url", type=str, default="http://localhost:8000/v1", help="URL of standard OpenAI completions API")
    parser.add_argument("--api_key", type=str, default=None, help="Optional API key for authorization")
    parser.add_argument("--tasks_dir", type=str, default="data/searchqa_debug/tasks", help="Tasks folder path")
    parser.add_argument("--corpus_dir", type=str, default="data/searchqa_debug/corpus", help="Corpus folder path")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_zero_shot", help="Output trajectory path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per step")
    args = parser.parse_args()

    print("=" * 60)
    print("ResearchAgent-RL: Qwen Zero-shot Rollout Loop")
    print(f"API Target: {args.model_url}")
    print("=" * 60)

    # 1. Load corpus & tasks
    corpus = CorpusStore(args.corpus_dir)
    corpus.load()
    task_files = sorted([os.path.join(args.tasks_dir, f) for f in os.listdir(args.tasks_dir) if f.endswith(".json")])
    tasks = [load_task(tf) for tf in task_files]
    print(f"Loaded {len(corpus)} corpus chunks and {len(tasks)} tasks.")

    # 2. Setup env & client
    env = ResearchEnv(corpus=corpus, max_steps=10)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(AnswerTool())
    
    client = LLMClient(api_url=args.model_url, api_key=args.api_key)
    tokenizer = MockTokenizer()

    # 3. Rollout loop
    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] Rollout Task: {task.task_id} - '{task.user_query}'")
        
        obs = env.reset(task)
        collector = ConversationCollector()
        collector.add_user_message(task.user_query)

        done = False
        step_count = 0

        try:
            while not done:
                # 3a. Get full text context up to start of assistant generation
                prompt = collector.get_prompt_for_generation()
                
                # 3b. Query Qwen model
                # Standard stop tokens to avoid endless generation beyond action tags
                raw_response = client.generate(
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop_tokens=["<|im_end|>", "</action>"]
                )
                
                # Suffix check: SGLang/vLLM raw stop tokens sometimes omit the stop token itself
                if not raw_response.endswith("</action>") and "<action>" in raw_response:
                    # Append stop token if truncated by stop parameter
                    raw_response += "</action>"

                # 3c. Parse action
                action = ActionParser.parse(raw_response)
                print(f"  Step {step_count}: Action={action.tool}, Action status={'SUCCESS' if action.tool != 'INVALID' else 'INVALID'}")
                
                # 3d. Step environment
                obs, done, reason = env.step(action)
                step_count += 1

                # 3e. Record response
                collector.add_assistant_response(raw_response)

                # 3f. Record observation
                if not done:
                    last_step = env._state.trajectory[-1]
                    success = False
                    error = last_step.error_message
                    data = {}
                    if last_step.is_valid and last_step.tool_result is not None:
                        success = last_step.tool_result.success
                        if not success:
                            error = last_step.tool_result.error
                        else:
                            data = last_step.tool_result.data
                    collector.add_observation(action.tool, success, data, error)

            episode_result = env.finalize_episode()
            metrics = compute_metrics(
                episode_result.final_answer,
                task.ground_truth_answer,
                episode_result.cited_chunk_ids,
                task.ground_truth_citations
            )

            # 4. Build and visualize loss mask
            input_ids, loss_mask = build_loss_mask(collector.segments, tokenizer)
            
            print(f"  Done reason: {episode_result.done_reason}")
            print(f"  Metrics: EM={metrics['exact_match']:.2f}, Contains={metrics['contains']:.2f}, Token F1={metrics['token_f1']:.2f}, Citation F1={metrics['citation_f1']:.2f}")

            if idx == 1:
                visualize_mask(input_ids, loss_mask)

            # Save outputs
            output_path = os.path.join(args.output_dir, f"{task.task_id}_qwen_zero_shot.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "task": task.to_dict(),
                    "episode": episode_result.to_dict(),
                    "full_conversation_text": collector.get_full_text(),
                    "metrics": metrics,
                    "token_mask_details": {
                        "seq_len": len(input_ids),
                        "trainable_tokens": sum(loss_mask),
                        "loss_mask_percentage": f"{(sum(loss_mask)/len(input_ids))*100:.2f}%"
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"  Saved trajectory to {output_path}")
            
            results.append(metrics)
            
        except Exception as e:
            print(f"  [ERROR] Failed to run rollout for task {task.task_id}: {e}", file=sys.stderr)

    if results:
        avg_em = sum(r["exact_match"] for r in results) / len(results)
        avg_contains = sum(r["contains"] for r in results) / len(results)
        avg_token_f1 = sum(r["token_f1"] for r in results) / len(results)
        avg_cit_f1 = sum(r["citation_f1"] for r in results) / len(results)
        print("\n" + "=" * 60)
        print("Zero-shot Rollout Loop Complete! Averages:")
        print(f"  Average Answer EM (Strict): {avg_em:.2f}")
        print(f"  Average Answer Contains: {avg_contains:.2f}")
        print(f"  Average Answer Token F1: {avg_token_f1:.2f}")
        print(f"  Average Citation F1: {avg_cit_f1:.2f}")
        print("=" * 60)

if __name__ == "__main__":
    main()
