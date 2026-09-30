from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, Any, List

from research_agent.core.env.env import ResearchEnv
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.schema.task import TaskSample, Rubric, TaskType
from research_agent.core.schema.parser import ActionParser
from research_agent.core.env.llm_client import LLMClient
from research_agent.core.env.rollout_collector import ConversationCollector
from research_agent.core.env.mask_builder import MockTokenizer
from research_agent.core.tools.search import SearchTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.answer import AnswerTool
from research_agent.adapters.slime.custom_reward import compute_metrics

class LocalModelInference:
    """Helper class to run offline inference via transformers if no server is running."""
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading local model and tokenizer from: {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto", 
            trust_remote_code=True
        )
        print("Local model loaded successfully.")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0, stop_tokens: list = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Configure stop criteria
        eos_token_id = self.tokenizer.eos_token_id
        
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "temperature": temperature if temperature > 0.0 else None,
            "top_p": top_p if temperature > 0.0 else None,
            "eos_token_id": eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or eos_token_id
        }
        # Filter out None values
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        
        outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response


def run_evaluation():
    parser = argparse.ArgumentParser(description="Zero-shot Tool-use Baseline Evaluator for HotpotQA")
    parser.add_argument("--tasks_dir", type=str, required=True, help="Directory containing test task JSON files")
    parser.add_argument("--corpus_dir", type=str, required=True, help="Directory containing wiki corpus chunks")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Local path to model (for local inference)")
    parser.add_argument("--actor_model_url", type=str, default=None, help="OpenAI-compatible server URL (vLLM/SGLang)")
    parser.add_argument("--model", type=str, default="Qwen3.5-9B-Instruct", help="Model identifier to pass to server")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation reports")
    parser.add_argument("--max_steps", type=int, default=6, help="Maximum turns for the agent environment")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p sampling config")
    parser.add_argument("--smoke", action="store_true", help="Run only a 10-task subset for quick smoke test validation")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("STARTING HOTPOTQA BASELINE EVALUATION")
    print("=" * 60)

    # 1. Setup outputs
    os.makedirs(args.output_dir, exist_ok=True)
    per_task_file = os.path.join(args.output_dir, "per_task.jsonl")
    summary_file = os.path.join(args.output_dir, "metrics_summary.json")

    # 2. Setup database and tools
    print(f"Loading corpus store from: {args.corpus_dir} ...")
    corpus = CorpusStore(args.corpus_dir)
    corpus.load()
    
    # 3. Setup Inference Engine
    inference_engine = None
    tokenizer = None
    
    if args.actor_model_url:
        print(f"Using remote server endpoint: {args.actor_model_url}")
        client = LLMClient(api_url=args.actor_model_url)
    elif args.model_name_or_path:
        print(f"Using local transformers model: {args.model_name_or_path}")
        inference_engine = LocalModelInference(args.model_name_or_path)
        tokenizer = inference_engine.tokenizer
    else:
        print("[ERROR] Must provide either --actor_model_url or --model_name_or_path")
        sys.exit(1)

    if tokenizer is None:
        if args.model_name_or_path and os.path.exists(args.model_name_or_path):
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            except Exception:
                tokenizer = MockTokenizer()
        else:
            tokenizer = MockTokenizer()

    # 4. Load tasks
    task_files = sorted([f for f in os.listdir(args.tasks_dir) if f.endswith(".json")])
    if not task_files:
        print(f"[ERROR] No task files found in {args.tasks_dir}")
        sys.exit(1)

    if args.smoke:
        print("Running in SMOKE mode. Limiting evaluation to first 10 tasks.")
        task_files = task_files[:10]

    print(f"Loaded {len(task_files)} tasks for evaluation.")

    all_records = []
    
    # 5. Run evaluation loop
    for idx, fname in enumerate(task_files):
        task_path = os.path.join(args.tasks_dir, fname)
        with open(task_path, "r", encoding="utf-8") as f:
            task_dict = json.load(f)
            
        task = TaskSample(
            task_id=task_dict["task_id"],
            task_type=TaskType.SURVEY_SYNTHESIS,
            user_query=task_dict["user_query"],
            rubric=Rubric(),
            ground_truth_answer=task_dict["ground_truth_answer"],
            ground_truth_citations=task_dict["ground_truth_citations"],
            reference_docs=task_dict.get("reference_docs", [])
        )
        
        print(f"[{idx+1}/{len(task_files)}] Running task: {task.task_id} ...")
        
        # Fresh local env for this task
        env = ResearchEnv(corpus=corpus, max_steps=args.max_steps)
        env.register_tool(SearchTool())
        env.register_tool(ReadTool())
        env.register_tool(AnswerTool())
        
        obs = env.reset(task)
        collector = ConversationCollector()
        collector.add_user_message(task.user_query)
        
        prompt_text = collector.segments[0].to_chatml() + collector.segments[1].to_chatml()
        prompt_tokens_count = len(tokenizer.encode(prompt_text))
        
        done = False
        step_count = 0
        total_latency = 0.0
        response_tokens_count = 0
        
        # Trajectory stats
        parse_error_count = 0
        tool_error_count = 0
        invalid_action_count = 0
        search_count = 0
        read_count = 0
        answer_count = 0

        # Execute rollout loop
        while not done:
            prompt = collector.get_prompt_for_generation()
            
            start_time = time.time()
            if args.actor_model_url:
                raw_response = client.generate(
                    prompt,
                    model=args.model,
                    max_tokens=256,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_tokens=["<|im_end|>", "</action>"]
                )
            else:
                raw_response = inference_engine.generate(
                    prompt,
                    max_tokens=256,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_tokens=["<|im_end|>", "</action>"]
                )
            latency = time.time() - start_time
            total_latency += latency
            
            response_tokens_count += len(tokenizer.encode(raw_response))
            
            # Post-process missing action wrapper tags
            if not raw_response.endswith("</action>") and "<action>" in raw_response:
                raw_response += "</action>"

            # Parse action
            action = ActionParser.parse(raw_response)
            
            # Increment parsing issue stats
            if "<action>" not in raw_response or "</action>" not in raw_response:
                parse_error_count += 1
            
            # Step environment
            obs, done, reason = env.step(action)
            step_count += 1
            
            # Record metrics per step
            if not env._state.trajectory[-1].is_valid:
                invalid_action_count += 1
            else:
                if action.tool == "SEARCH":
                    search_count += 1
                elif action.tool == "READ":
                    read_count += 1
                elif action.tool == "ANSWER":
                    answer_count += 1
                
                last_step = env._state.trajectory[-1]
                if last_step.tool_result and not last_step.tool_result.success:
                    tool_error_count += 1

            collector.add_assistant_response(raw_response)
            
            if not done:
                last_step = env._state.trajectory[-1]
                success = last_step.is_valid and last_step.tool_result is not None and last_step.tool_result.success
                error = last_step.error_message if not last_step.is_valid else (last_step.tool_result.error if last_step.tool_result else "")
                data = last_step.tool_result.data if (last_step.tool_result and last_step.tool_result.success) else {}
                collector.add_observation(action.tool, success, data, error)

        # Finalize
        episode_result = env.finalize_episode()
        metrics = compute_metrics(
            episode_result.final_answer, 
            task.ground_truth_answer, 
            episode_result.cited_chunk_ids, 
            task.ground_truth_citations
        )

        # Calculate exact reward splits
        contains_score = metrics.get("contains", 0.0)
        token_f1 = metrics.get("token_f1", 0.0)
        citation_f1 = metrics.get("citation_f1", 0.0)
        
        answer_reward = (1.0 * contains_score) + (0.5 * token_f1)
        citation_reward = 0.5 * citation_f1
        step_penalty = -0.01 * step_count
        invalid_penalty = -0.05 * env._state.invalid_action_count
        total_reward = max(-2.0, min(2.0, answer_reward + citation_reward + step_penalty + invalid_penalty))

        # Precision & Recall calculations
        cited_set = set(episode_result.cited_chunk_ids)
        gt_set = set(task.ground_truth_citations)
        intersection = cited_set & gt_set
        precision = len(intersection) / len(cited_set) if cited_set else 0.0
        recall = len(intersection) / len(gt_set) if gt_set else 0.0

        # Construct task record
        record = {
            "task_id": task.task_id,
            "user_query": task.user_query,
            "ground_truth_answer": task.ground_truth_answer,
            "ground_truth_citations": task.ground_truth_citations,
            "final_answer": episode_result.final_answer,
            "cited_chunk_ids": episode_result.cited_chunk_ids,
            "done_reason": episode_result.done_reason,
            "steps_count": step_count,
            "invalid_action_count": env._state.invalid_action_count,
            "parse_error_count": parse_error_count,
            "tool_error_count": tool_error_count,
            "search_count": search_count,
            "read_count": read_count,
            "answer_count": answer_count,
            "answer_exact_match": metrics.get("exact_match", 0.0),
            "answer_contains": contains_score,
            "answer_token_f1": token_f1,
            "citation_precision": precision,
            "citation_recall": recall,
            "citation_f1": citation_f1,
            "answer_reward": answer_reward,
            "citation_reward": citation_reward,
            "step_penalty": step_penalty,
            "invalid_penalty": invalid_penalty,
            "total_reward": total_reward,
            "latency_sec": total_latency,
            "prompt_tokens": prompt_tokens_count,
            "response_tokens": response_tokens_count,
            "trajectory": episode_result.to_dict()["trajectory"]
        }
        
        all_records.append(record)
        
        # Save progress incrementally to jsonl
        with open(per_task_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 6. Compute summary aggregates
    n_tasks = len(all_records)
    
    # Aggregated metrics calculation
    summaries = {
        # Answer metrics
        "answer_exact_match_mean": float(np.mean([r["answer_exact_match"] for r in all_records])),
        "answer_contains_mean": float(np.mean([r["answer_contains"] for r in all_records])),
        "answer_token_f1_mean": float(np.mean([r["answer_token_f1"] for r in all_records])),
        "answer_non_empty_rate": float(np.mean([1.0 if len(r["final_answer"].strip()) > 0 else 0.0 for r in all_records])),

        # Citation metrics
        "citation_precision_mean": float(np.mean([r["citation_precision"] for r in all_records])),
        "citation_recall_mean": float(np.mean([r["citation_recall"] for r in all_records])),
        "citation_f1_mean": float(np.mean([r["citation_f1"] for r in all_records])),
        "citation_exists_rate": float(np.mean([1.0 if len(r["cited_chunk_ids"]) > 0 else 0.0 for r in all_records])),
        "support_coverage_rate": float(np.mean([1.0 if set(r["ground_truth_citations"]).issubset(set(r["cited_chunk_ids"])) else 0.0 for r in all_records])),

        # Tool metrics
        "format_valid_rate": float(np.mean([1.0 if r["parse_error_count"] == 0 else 0.0 for r in all_records])),
        "parse_valid_rate": float(np.mean([1.0 if r["parse_error_count"] == 0 else 0.0 for r in all_records])),
        "action_valid_rate": float(np.mean([1.0 if r["invalid_action_count"] == 0 else 0.0 for r in all_records])),
        "tool_success_rate": float(np.mean([1.0 - (r["tool_error_count"] / r["steps_count"]) if r["steps_count"] > 0 else 1.0 for r in all_records])),
        "search_success_rate": float(np.mean([1.0 if r["search_count"] > 0 else 0.0 for r in all_records])),
        "read_success_rate": float(np.mean([1.0 if r["read_count"] > 0 else 0.0 for r in all_records])),
        "answer_submit_rate": float(np.mean([1.0 if r["answer_count"] > 0 else 0.0 for r in all_records])),
        "invalid_action_per_task": float(np.mean([r["invalid_action_count"] for r in all_records])),
        "parse_error_per_task": float(np.mean([r["parse_error_count"] for r in all_records])),
        "tool_error_per_task": float(np.mean([r["tool_error_count"] for r in all_records])),

        # Trajectory metrics
        "avg_steps": float(np.mean([r["steps_count"] for r in all_records])),
        "avg_search_count": float(np.mean([r["search_count"] for r in all_records])),
        "avg_read_count": float(np.mean([r["read_count"] for r in all_records])),
        "avg_answer_count": float(np.mean([r["answer_count"] for r in all_records])),
        "max_steps_rate": float(np.mean([1.0 if r["done_reason"] == "max_steps" else 0.0 for r in all_records])),
        "answer_submitted_rate": float(np.mean([1.0 if r["done_reason"] == "answer_submitted" else 0.0 for r in all_records])),
        "no_progress_rate": float(np.mean([1.0 if r["done_reason"] == "no_progress_exceeded" else 0.0 for r in all_records])),

        # Reward metrics
        "answer_reward_mean": float(np.mean([r["answer_reward"] for r in all_records])),
        "citation_reward_mean": float(np.mean([r["citation_reward"] for r in all_records])),
        "step_penalty_mean": float(np.mean([r["step_penalty"] for r in all_records])),
        "invalid_penalty_mean": float(np.mean([r["invalid_penalty"] for r in all_records])),
        "total_reward_mean": float(np.mean([r["total_reward"] for r in all_records])),
        "total_reward_std": float(np.std([r["total_reward"] for r in all_records])),

        # Efficiency metrics
        "avg_latency_sec": float(np.mean([r["latency_sec"] for r in all_records])),
        "p50_latency_sec": float(np.percentile([r["latency_sec"] for r in all_records], 50)),
        "p95_latency_sec": float(np.percentile([r["latency_sec"] for r in all_records], 95)),
        "avg_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in all_records])),
        "avg_response_tokens": float(np.mean([r["response_tokens"] for r in all_records])),
        "avg_total_tokens": float(np.mean([r["prompt_tokens"] + r["response_tokens"] for r in all_records])),

        # Failure cause distribution
        "failure_parse_error_count": int(sum(1 for r in all_records if r["parse_error_count"] > 0)),
        "failure_invalid_action_count": int(sum(1 for r in all_records if r["invalid_action_count"] > 0)),
        "failure_no_search_result_count": int(sum(1 for r in all_records if r["search_count"] == 0)),
        "failure_read_missing_chunk_count": int(sum(1 for r in all_records if r["tool_error_count"] > 0 and r["read_count"] > 0)),
        "failure_answer_no_citation_count": int(sum(1 for r in all_records if len(r["cited_chunk_ids"]) == 0 and r["answer_count"] > 0)),
        "failure_wrong_answer_count": int(sum(1 for r in all_records if r["answer_contains"] == 0.0)),
        "failure_max_steps_count": int(sum(1 for r in all_records if r["done_reason"] == "max_steps"))
    }

    # Save summary
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"- Task records saved to: {per_task_file}")
    print(f"- Summary report saved to: {summary_file}")
    print("=" * 60)
    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    run_evaluation()
