from __future__ import annotations
import os
import json

def main():
    tasks_dir = "data/searchqa_debug/tasks"
    output_file = "data/searchqa_debug/rollout_dataset.jsonl"
    
    if not os.path.exists(tasks_dir):
        print(f"[ERROR] Tasks directory {tasks_dir} does not exist.")
        return
        
    tasks = []
    for name in sorted(os.listdir(tasks_dir)):
        if name.endswith(".json"):
            file_path = os.path.join(tasks_dir, name)
            with open(file_path, "r", encoding="utf-8") as f:
                task = json.load(f)
                
                # Format to match Slime's Sample initialization from JSONL
                sample = {
                    "prompt": task["user_query"],
                    "metadata": {
                        "task_id": task["task_id"],
                        "user_query": task["user_query"],
                        "ground_truth_answer": task["ground_truth_answer"],
                        "ground_truth_citations": task["ground_truth_citations"],
                        "reference_docs": task.get("reference_docs", [])
                    }
                }
                tasks.append(sample)
                
    with open(output_file, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated Slime dataset with {len(tasks)} items at: {output_file}")

if __name__ == "__main__":
    main()
