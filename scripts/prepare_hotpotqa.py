from __future__ import annotations
import os
import json
import random
from typing import Dict, Any, List

# Target files and directory configurations
DATA_DIR = "data"
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
TRAIN_TASKS_DIR = os.path.join(DATA_DIR, "tasks", "train")
TEST_TASKS_DIR = os.path.join(DATA_DIR, "tasks", "test")

def main():
    print("============================================================")
    print("ResearchAgent-RL: HotpotQA Benchmark Dataset Preparer")
    print("============================================================")

    # Ensure output directories exist
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(TRAIN_TASKS_DIR, exist_ok=True)
    os.makedirs(TEST_TASKS_DIR, exist_ok=True)

    # Use Chinese HF mirror endpoint to bypass server-side network restrictions
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library is not installed.")
        print("Please install it first: python3 -m pip install datasets")
        return

    print("Loading HotpotQA validation split (distractor setting) from HF Mirror...")
    # Loading validation split because it is moderately sized and clean
    raw_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Select 1000 tasks randomly (or as many as available)
    total_samples = min(len(raw_dataset), 1000)
    indices = random.sample(range(len(raw_dataset)), total_samples)
    samples = [raw_dataset[i] for i in indices]
    
    print(f"Successfully loaded and shuffled {len(samples)} samples.")

    # Split: 70% Training, 30% Testing
    train_size = int(len(samples) * 0.7)
    train_samples = samples[:train_size]
    test_samples = samples[train_size:]
    
    print(f"Splitting: {len(train_samples)} training tasks, {len(test_samples)} testing tasks.")

    processed_corpus_count = 0

    def process_and_save_samples(dataset_split: List[Dict[str, Any]], target_dir: str):
        nonlocal processed_corpus_count
        for item in dataset_split:
            task_id = f"hotpot_{item['id']}"
            question = item["question"]
            answer = item["answer"]
            
            # Map Wiki paragraphs to corpus chunks
            context = item["context"]  # Format: list of [title, list of sentences]
            supporting_facts = item["supporting_facts"]  # Format: list of [title, sent_idx]
            supporting_titles = set(fact[0] for fact in supporting_facts)
            
            # Process wikipedia paragraphs in context
            citations = []
            for title, sentences in zip(context["title"], context["sentences"]):
                # Create a unique chunk ID using title and task_id to avoid crossover
                sanitized_title = "".join(c if c.isalnum() else "_" for c in title).lower()
                chunk_id = f"{sanitized_title}_{task_id}"
                text_content = "".join(sentences)
                
                # Write to global corpus database
                chunk_file = os.path.join(CORPUS_DIR, f"{chunk_id}.json")
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "chunk_id": chunk_id,
                        "title": title,
                        "text": text_content,
                        "metadata": {"task_source": "hotpot_qa"}
                    }, f, ensure_ascii=False, indent=2)
                processed_corpus_count += 1
                
                # If this paragraph is in supporting titles, add to ground truth citations
                if title in supporting_titles:
                    citations.append(chunk_id)

            # Define the task configuration file
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

    print("Processing and writing training tasks...")
    process_and_save_samples(train_samples, TRAIN_TASKS_DIR)
    
    print("Processing and writing testing tasks...")
    process_and_save_samples(test_samples, TEST_TASKS_DIR)

    print(f"\n[SUCCESS] Setup Completed!")
    print(f"- Total corpus documents created: {processed_corpus_count}")
    print(f"- Train tasks written to: {TRAIN_TASKS_DIR} ({len(train_samples)} files)")
    print(f"- Test tasks written to: {TEST_TASKS_DIR} ({len(test_samples)} files)")
    print("============================================================")

if __name__ == "__main__":
    main()
