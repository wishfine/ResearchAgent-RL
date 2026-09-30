from __future__ import annotations
import unittest
import os
import json

class TestSFTDatasetStructure(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "outputs/sft_dataset.json"

    def test_sft_dataset_file_exists_and_valid(self):
        # Verify the file is generated
        self.assertTrue(os.path.exists(self.dataset_path), "SFT dataset file does not exist. Run scripts/generate_sft_dataset.py first.")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertEqual(len(data), 10, "SFT dataset should contain exactly 10 tasks.")
        
        for item in data:
            self.assertIn("task_id", item)
            self.assertIn("messages", item)
            
            messages = item["messages"]
            self.assertGreaterEqual(len(messages), 3, "Each task dialog should have at least 3 messages (System, User, Assistant).")
            
            # Check role sequences: system -> user -> assistant -> user -> assistant etc.
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            
            for idx, msg in enumerate(messages):
                role = msg["role"]
                content = msg["content"]
                
                # Check valid roles only (OpenAI compatible)
                self.assertIn(role, {"system", "user", "assistant"}, f"Invalid role: {role}. Only system, user, assistant allowed.")
                self.assertIsNotNone(content)
                self.assertGreater(len(content.strip()), 0)
                
                # Check turn rotation
                if idx > 0:
                    prev_role = messages[idx-1]["role"]
                    if prev_role == "user":
                        self.assertEqual(role, "assistant")
                    elif prev_role == "assistant":
                        self.assertEqual(role, "user")
                
                # Check assistant output format: reasoning + action tags
                if role == "assistant":
                    self.assertTrue(content.startswith("<reasoning>"), f"Assistant content should start with reasoning tags: {content[:40]}...")
                    self.assertIn("</reasoning>", content)
                    self.assertIn("<action>", content)
                    self.assertTrue(content.endswith("</action>"), f"Assistant content should end with action tags: ...{content[-40:]}")

if __name__ == "__main__":
    unittest.main()
