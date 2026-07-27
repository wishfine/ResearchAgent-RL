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
from research_agent.core.env.rollout_collector import ConversationCollector
from research_agent.core.tools.search import SearchTool
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.answer import AnswerTool

ANSWERS_MAP = {
    "toy_task_001": ("Canada", "maple_leaf_flag"),
    "toy_task_002": ("William Shakespeare", "romeo_juliet_author"),
    "toy_task_003": ("Paris", "france_capital"),
    "toy_task_004": ("Mars", "mars_red_planet"),
    "toy_task_005": ("Albert Einstein", "general_relativity_creator"),
    "toy_task_006": ("H2O", "water_chemical_symbol"),
    "toy_task_007": ("Mount Everest", "tallest_mountain"),
    "toy_task_008": ("Pacific Ocean", "largest_ocean"),
    "toy_task_009": ("Leonardo da Vinci", "mona_lisa_painter"),
    "toy_task_010": ("1945", "ww2_end_year")
}

class OracleTeacherPolicy:
    def decide(self, obs: Any) -> str:
        tid = obs.task_id
        ans, correct_chunk_id = ANSWERS_MAP.get(tid, ("Unknown", ""))
        
        # Step 0: SEARCH
        if "SEARCH" not in obs.trajectory:
            return (
                f"<reasoning>To answer '{obs.user_query}', I need to search the database first.</reasoning>\n"
                f"<action>\n"
                f"{{\n"
                f"  \"tool\": \"SEARCH\",\n"
                f"  \"params\": {{\"query\": \"{obs.user_query}\", \"topk\": 3}}\n"
                f"}}\n"
                f"</action>"
            )
        # Step 1: READ
        if "READ" not in obs.trajectory:
            return (
                f"<reasoning>I found relevant documents. I will read the content of chunk {correct_chunk_id} to extract evidence.</reasoning>\n"
                f"<action>\n"
                f"{{\n"
                f"  \"tool\": \"READ\",\n"
                f"  \"params\": {{\"chunk_ids\": [\"{correct_chunk_id}\"]}}\n"
                f"}}\n"
                f"</action>"
            )
        # Step 2: ANSWER
        return (
            f"<reasoning>I have read the document chunk {correct_chunk_id} and verified the fact. The answer is {ans}. I will submit the answer with citation.</reasoning>\n"
            f"<action>\n"
            f"{{\n"
            f"  \"tool\": \"ANSWER\",\n"
            f"  \"params\": {{\"answer_text\": \"The answer is {ans}.\", \"cited_chunk_ids\": [\"{correct_chunk_id}\"]}}\n"
            f"}}\n"
            f"</action>"
        )

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

def main():
    print("=" * 60)
    print("ResearchAgent-RL: SFT Cold-start Dataset Generation")
    print("=" * 60)

    # 1. Setup paths
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    corpus_dir = os.path.join(base_dir, "data", "searchqa_debug", "corpus")
    tasks_dir = os.path.join(base_dir, "data", "searchqa_debug", "tasks")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load corpus & tasks
    corpus = CorpusStore(corpus_dir)
    corpus.load()
    task_files = sorted([os.path.join(tasks_dir, f) for f in os.listdir(tasks_dir) if f.endswith(".json")])
    tasks = [load_task(tf) for tf in task_files]

    # 3. Setup env & oracle policy
    env = ResearchEnv(corpus=corpus, max_steps=10)
    env.register_tool(SearchTool())
    env.register_tool(ReadTool())
    env.register_tool(AnswerTool())
    
    policy = OracleTeacherPolicy()

    # 4. Generate trajectories
    sft_dataset = []

    for idx, task in enumerate(tasks, 1):
        print(f"Generating SFT rollout {idx}/{len(tasks)} for task {task.task_id}...")
        
        obs = env.reset(task)
        collector = ConversationCollector()
        collector.add_user_message(task.user_query)

        done = False
        while not done:
            raw_response = policy.decide(obs)
            action = ActionParser.parse(raw_response)
            obs, done, reason = env.step(action)

            # Record assistant turn
            collector.add_assistant_response(raw_response)

            # Record observation turn if not done
            if not done:
                last_step = env._state.trajectory[-1]
                success = last_step.is_valid and last_step.tool_result is not None and last_step.tool_result.success
                error = last_step.error_message if not last_step.is_valid else (last_step.tool_result.error if last_step.tool_result else "")
                data = last_step.tool_result.data if (last_step.tool_result and last_step.tool_result.success) else {}
                collector.add_observation(action.tool, success, data, error)

        # Confirm success
        episode_result = env.finalize_episode()
        assert episode_result.done_reason == "answer_submitted", f"Oracle failed on task {task.task_id}!"
        
        # 5. Convert collector segments to standard messages JSON format
        dialog_messages = []
        for seg in collector.segments:
            role = "user" if seg.role in ("user", "observation") else seg.role
            dialog_messages.append({
                "role": role,
                "content": seg.text
            })
            
        sft_dataset.append({
            "task_id": task.task_id,
            "messages": dialog_messages
        })

    # Save to file
    output_path = os.path.join(output_dir, "sft_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 60)
    print(f"Successfully generated SFT dataset with {len(sft_dataset)} items!")
    print(f"Dataset path: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
