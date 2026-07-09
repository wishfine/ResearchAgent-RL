from __future__ import annotations
import unittest
import os
import json

class TestHotpotQAPreparation(unittest.TestCase):
    def test_debug_split_layout_and_stats(self):
        stats_path = "data/hotpotqa_debug/stats.json"
        if not os.path.exists(stats_path):
            self.skipTest("hotpotqa_debug stats.json not found. Run prepare_hotpotqa.py first.")
            
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            
        self.assertEqual(stats["split_name"], "hotpotqa_debug")
        self.assertEqual(stats["train_tasks_count"], 14)
        self.assertEqual(stats["test_tasks_count"], 6)
        self.assertEqual(stats["total_tasks"], 20)
        self.assertFalse(stats["overlap_verification"]["leakage_detected"])
        self.assertTrue(stats["overlap_verification"]["all_citations_exist"])
        
        # Verify that task configurations actually exist
        train_dir = "data/hotpotqa_debug/tasks/train"
        test_dir = "data/hotpotqa_debug/tasks/test"
        self.assertEqual(len(os.listdir(train_dir)), 14)
        self.assertEqual(len(os.listdir(test_dir)), 6)

    def test_mini_split_layout_and_stats(self):
        stats_path = "data/hotpotqa_mini/stats.json"
        if not os.path.exists(stats_path):
            self.skipTest("hotpotqa_mini stats.json not found. Run prepare_hotpotqa.py first.")
            
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            
        self.assertEqual(stats["split_name"], "hotpotqa_mini")
        self.assertEqual(stats["train_tasks_count"], 100)
        self.assertEqual(stats["test_tasks_count"], 30)
        self.assertEqual(stats["total_tasks"], 130)
        self.assertFalse(stats["overlap_verification"]["leakage_detected"])
        self.assertTrue(stats["overlap_verification"]["all_citations_exist"])
        
        # Verify that task configurations actually exist
        train_dir = "data/hotpotqa_mini/tasks/train"
        test_dir = "data/hotpotqa_mini/tasks/test"
        self.assertEqual(len(os.listdir(train_dir)), 100)
        self.assertEqual(len(os.listdir(test_dir)), 30)
        
        # Verify citation chunk file exists in corpus directory for random samples
        corpus_dir = "data/hotpotqa_mini/corpus"
        train_files = os.listdir(train_dir)
        if train_files:
            sample_file = os.path.join(train_dir, train_files[0])
            with open(sample_file, "r") as f:
                task_data = json.load(f)
            for citation_id in task_data["ground_truth_citations"]:
                citation_path = os.path.join(corpus_dir, f"{citation_id}.json")
                self.assertTrue(os.path.exists(citation_path), f"Citation {citation_id} file not found!")

if __name__ == "__main__":
    unittest.main()
