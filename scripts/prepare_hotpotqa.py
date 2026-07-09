from __future__ import annotations
import os
import json
import random
import sys
from typing import Dict, Any, List

def sanitize_title(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title).lower()

def build_split(samples: List[Dict[str, Any]], output_dir: str, train_count: int, test_count: int) -> Dict[str, Any]:
    print(f"\nBuilding split in: {output_dir}")
    
    corpus_dir = os.path.join(output_dir, "corpus")
    train_tasks_dir = os.path.join(output_dir, "tasks", "train")
    test_tasks_dir = os.path.join(output_dir, "tasks", "test")
    
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(train_tasks_dir, exist_ok=True)
    os.makedirs(test_tasks_dir, exist_ok=True)
    
    # 1. Slice and partition samples
    total_needed = train_count + test_count
    selected = samples[:total_needed]
    
    train_samples = selected[:train_count]
    test_samples = selected[train_count:]
    
    # 2. Leakage verification check (disjoint sets)
    train_ids = set(item["id"] for item in train_samples)
    test_ids = set(item["id"] for item in test_samples)
    assert train_ids.isdisjoint(test_ids), "CRITICAL: Train and test task IDs overlap!"

    total_chunks_created = 0
    task_chunks_stats = []

    def process_split(split_items: List[Dict[str, Any]], target_dir: str) -> List[Dict[str, Any]]:
        nonlocal total_chunks_created
        processed_tasks = []
        
        for item in split_items:
            task_id = f"hotpot_{item['id']}"
            question = item["question"]
            answer = item["answer"]
            
            # Map Wiki paragraphs to corpus chunks
            context = item["context"]
            supporting_facts = item["supporting_facts"]
            supporting_titles = set(fact[0] for fact in supporting_facts)
            
            citations = []
            local_chunks = 0
            
            for title, sentences in zip(context["title"], context["sentences"]):
                chunk_id = f"{sanitize_title(title)}_{task_id}"
                text_content = "".join(sentences)
                
                # Save chunk to corpus
                chunk_file = os.path.join(corpus_dir, f"{chunk_id}.json")
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "chunk_id": chunk_id,
                        "title": title,
                        "text": text_content,
                        "metadata": {"task_source": "hotpot_qa"}
                    }, f, ensure_ascii=False, indent=2)
                
                total_chunks_created += 1
                local_chunks += 1
                
                if title in supporting_titles:
                    citations.append(chunk_id)

            # Assert that every citation ID maps to a file that was just created
            for cid in citations:
                citation_path = os.path.join(corpus_dir, f"{cid}.json")
                assert os.path.exists(citation_path), f"ERROR: Citation file {citation_path} was not created!"
                
            task_config = {
                "task_id": task_id,
                "user_query": question,
                "ground_truth_answer": answer,
                "ground_truth_citations": citations,
                "reference_docs": []
            }
            
            task_file = os.path.join(target_dir, f"{task_id}.json")
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task_config, f, ensure_ascii=False, indent=2)
                
            task_chunks_stats.append(local_chunks)
            processed_tasks.append(task_config)
            
        return processed_tasks

    print(f"Processing {len(train_samples)} training tasks...")
    train_tasks = process_split(train_samples, train_tasks_dir)
    
    print(f"Processing {len(test_samples)} testing tasks...")
    test_tasks = process_split(test_samples, test_tasks_dir)

    # 3. Create stats.json summary
    avg_chunks = sum(task_chunks_stats) / len(task_chunks_stats) if task_chunks_stats else 0
    stats = {
        "split_name": os.path.basename(output_dir),
        "total_tasks": len(train_tasks) + len(test_tasks),
        "train_tasks_count": len(train_tasks),
        "test_tasks_count": len(test_tasks),
        "total_corpus_chunks": total_chunks_created,
        "average_chunks_per_task": avg_chunks,
        "overlap_verification": {
            "leakage_detected": not train_ids.isdisjoint(test_ids),
            "all_citations_exist": True
        }
    }
    
    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    print(f"Saved stats to: {stats_file}")
    return stats


def main():
    print("============================================================")
    print("ResearchAgent-RL: HotpotQA Phase 1 Benchmark Setup")
    print("============================================================")

    # Set HF endpoint to mirror
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library is not installed.")
        print("Please install it first: python3 -m pip install datasets")
        return

    print("Loading HotpotQA validation split (distractor setting) from HF Mirror...")
    raw_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # We need 20 tasks for debug split, and 130 tasks for mini split (150 tasks in total)
    total_needed = 150
    if len(raw_dataset) < total_needed:
        print(f"[ERROR] Dataset only has {len(raw_dataset)} items. Need at least {total_needed}.")
        return

    # Deterministic shuffle to make it reproducible across runs
    indices = list(range(len(raw_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    selected_indices = indices[:total_needed]
    samples = [raw_dataset[i] for i in selected_indices]

    # Partition: first 20 for debug, next 130 for mini (no overlap between splits)
    debug_samples = samples[:20]
    mini_samples = samples[20:]

    # Build HotpotQA Debug: 14 Train, 6 Test
    debug_stats = build_split(debug_samples, "data/hotpotqa_debug", train_count=14, test_count=6)
    
    # Build HotpotQA Mini: 100 Train, 30 Test
    mini_stats = build_split(mini_samples, "data/hotpotqa_mini", train_count=100, test_count=30)

    print("\n[SUCCESS] Phase 1 HotpotQA Benchmark Generation Completed!")
    print(f"- Debug split tasks: {debug_stats['train_tasks_count']} train / {debug_stats['test_tasks_count']} test")
    print(f"- Mini split tasks: {mini_stats['train_tasks_count']} train / {mini_stats['test_tasks_count']} test")
    print("============================================================")

if __name__ == "__main__":
    main()
