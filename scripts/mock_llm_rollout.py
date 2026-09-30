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
from research_agent.core.schema.parser import ActionParser
from research_agent.core.schema.mock_policy import MockLLMPolicy
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

def visualize_mask(input_ids: list[int], loss_mask: list[int], tokenizer: MockTokenizer, max_len: int = 150):
    """Prints a visual token map showing which tokens are trainable [1] vs masked [0]."""
    print("\nToken Loss Mask Visualization (showing first 150 chars/tokens):")
    print("-" * 80)
    for idx in range(min(len(input_ids), max_len)):
        char = chr(input_ids[idx])
        mask = loss_mask[idx]
        
        # Format char for printing
        char_rep = repr(char)[1:-1]
        if char == "\n":
            char_rep = "\\n"
        
        # Color coding: Green for trainable [1], grey for masked [0]
        if mask == 1:
            # Underline / Highlight green
            print(f"\033[92m{char_rep}\033[0m", end="")
        else:
            print(f"\033[90m{char_rep}\033[0m", end="")
    print("\n" + "-" * 80)
    print("Color guide: \033[92mGreen (Trainable: loss_mask=1)\033[0m | \033[90mGrey (Masked: loss_mask=0)\033[0m")

def main():
    print("=" * 60)
    print("ResearchAgent-RL Milestone 0.5: Mock LLM Rollout Run")
    print("=" * 60)

    # 1. Setup paths
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    corpus_dir = os.path.join(base_dir, "data", "searchqa_debug", "corpus")
    tasks_dir = os.path.join(base_dir, "data", "searchqa_debug", "tasks")
    output_dir = os.path.join(base_dir, "outputs", "mock_rollout")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load corpus & tasks
    corpus = CorpusStore(corpus_dir)
    corpus.load()
    task_files = sorted([os.path.join(tasks_dir, f) for f in os.listdir(tasks_dir) if f.endswith(".json")])
    tasks = [load_task(tf) for tf in task_files]

    # 3. Setup env & policy
    env = ResearchEnv(corpus=corpus, max_steps=10)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(AnswerTool())
    
    policy = MockLLMPolicy()
    tokenizer = MockTokenizer()

    # 4. Rollout loop
    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/10] Rollout Task: {task.task_id} - '{task.user_query}'")
        
        # Reset env & initialize conversation collector
        obs = env.reset(task)
        collector = ConversationCollector()
        collector.add_user_message(task.user_query)

        done = False
        step_count = 0

        while not done:
            # 4a. Get the prompt that would be fed to the LLM (for validation)
            prompt_context = collector.get_prompt_for_generation()
            
            # 4b. Policy generates raw text response (simulating LLM)
            raw_response = policy.decide(obs)
            
            # 4c. Parse raw response to action
            action = ActionParser.parse(raw_response)
            
            print(f"  Step {step_count}: Action={action.tool}, Parse status={'SUCCESS' if action.tool != 'INVALID' else 'INVALID'}")
            
            # 4d. Step environment
            obs, done, reason = env.step(action)
            step_count += 1

            # 4e. Record assistant response in chat logs
            collector.add_assistant_response(raw_response)

            # 4f. Record observation in chat logs if not done
            if not done:
                # Find the last step details for observation logging
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

        # Finalize episode and generate metrics
        episode_result = env.finalize_episode()
        
        # 5. Build token-level Loss Mask
        input_ids, loss_mask = build_loss_mask(collector.segments, tokenizer)
        
        # Verify sizes match
        assert len(input_ids) == len(loss_mask), "input_ids and loss_mask size mismatch!"
        
        # Visualize masking for the first task as a demonstration
        if idx == 1:
            print("\nConversation Flow:")
            print(collector.get_full_text())
            visualize_mask(input_ids, loss_mask, tokenizer, max_len=400)

        # Save trajectory & prompt mask logs
        output_path = os.path.join(output_dir, f"{task.task_id}_mock_rollout.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": task.to_dict(),
                "episode": episode_result.to_dict(),
                "full_conversation_text": collector.get_full_text(),
                "token_mask_details": {
                    "seq_len": len(input_ids),
                    "trainable_tokens": sum(loss_mask),
                    "masked_tokens": len(input_ids) - sum(loss_mask),
                    "loss_mask_percentage": f"{(sum(loss_mask)/len(input_ids))*100:.2f}%"
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved rollout outputs to {output_path}")

    print("\n" + "=" * 60)
    print("Milestone 0.5: Mock LLM Rollout Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
