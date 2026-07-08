from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock
import asyncio
import json

from research_agent.adapters.slime.custom_generator import custom_generate, SlimeSample
from research_agent.adapters.slime.custom_reward import custom_rm

class SlimeMockArgs:
    def __init__(self):
        self.actor_model_url = "http://localhost:8000/v1"
        self.corpus_dir = "data/searchqa_debug/corpus"
        self.max_steps = 5
        self.reward_type = "token_f1"
        self.invalid_penalty = 0.05
        self.step_penalty = 0.01

class SlimeMockInputSample:
    def __init__(self):
        self.task_id = "toy_task_001"
        self.user_query = "Which country's flag features a red maple leaf?"
        self.ground_truth_answer = "Canada"
        self.ground_truth_citations = ["maple_leaf_flag"]


class TestSlimeAdapter(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.args = SlimeMockArgs()
        self.input_sample = SlimeMockInputSample()
        self.sampling_params = {"max_tokens": 100, "temperature": 0.0}

    def tearDown(self):
        self.loop.close()

    @patch("urllib.request.urlopen")
    def test_custom_generator_end_to_end(self, mock_urlopen):
        # We mock a sequence of three HTTP responses:
        # 1. Search action response
        # 2. Read action response
        # 3. Answer action response
        
        mock_resp_1 = MagicMock()
        mock_resp_1.read.return_value = json.dumps({
            "choices": [{"text": '<reasoning>search</reasoning><action>{"tool": "SEARCH", "params": {"query": "Canada maple leaf", "topk": 1}}</action>'}]
        }).encode("utf-8")
        
        mock_resp_2 = MagicMock()
        mock_resp_2.read.return_value = json.dumps({
            "choices": [{"text": '<reasoning>read</reasoning><action>{"tool": "READ", "params": {"chunk_ids": ["maple_leaf_flag"]}}</action>'}]
        }).encode("utf-8")

        mock_resp_3 = MagicMock()
        mock_resp_3.read.return_value = json.dumps({
            "choices": [{"text": '<reasoning>answer</reasoning><action>{"tool": "ANSWER", "params": {"answer_text": "The answer is Canada.", "cited_chunk_ids": ["maple_leaf_flag"]}}</action>'}]
        }).encode("utf-8")

        mock_enter_1 = MagicMock()
        mock_enter_1.__enter__.return_value = mock_resp_1
        
        mock_enter_2 = MagicMock()
        mock_enter_2.__enter__.return_value = mock_resp_2
        
        mock_enter_3 = MagicMock()
        mock_enter_3.__enter__.return_value = mock_resp_3

        mock_urlopen.side_effect = [mock_enter_1, mock_enter_2, mock_enter_3]

        # Execute custom_generate asynchronously
        result = self.loop.run_until_complete(
            custom_generate(self.args, self.input_sample, self.sampling_params)
        )

        self.assertIsInstance(result, SlimeSample)
        self.assertEqual(result.task_id, "toy_task_001")
        
        # Verify metadata is correct
        self.assertEqual(result.metadata["done_reason"], "answer_submitted")
        self.assertEqual(result.metadata["steps_count"], 3)
        self.assertEqual(result.metadata["invalid_action_count"], 0)
        self.assertEqual(result.metadata["final_answer"], "The answer is Canada.")
        self.assertEqual(result.metadata["cited_chunk_ids"], ["maple_leaf_flag"])

        # Check token lists matching sizes
        self.assertEqual(len(result.input_ids), len(result.loss_mask))
        
        # Verify that prompt is exactly system + user
        self.assertTrue(result.prompt.startswith("<|im_start|>system"))
        self.assertTrue(result.prompt.endswith("features a red maple leaf?<|im_end|>\n"))

    def test_custom_reward_success(self):
        sample = {
            "ground_truth_answer": "Canada",
            "ground_truth_citations": ["maple_leaf_flag"],
            "metadata": {
                "final_answer": "Canada",
                "cited_chunk_ids": ["maple_leaf_flag"],
                "steps_count": 3,
                "invalid_action_count": 0
            }
        }
        # Run custom_rm
        reward = self.loop.run_until_complete(custom_rm(self.args, sample))
        
        # Expect maximum base score: 1.0 (contains) + 0.5 (token_f1) + 0.5 (citation_f1) = 2.0
        # Deductions: 3 steps * 0.01 = 0.03
        # Expected: 2.0 - 0.03 = 1.97
        self.assertAlmostEqual(reward, 1.97, places=4)

    def test_custom_reward_partial_match(self):
        sample = {
            "ground_truth_answer": "Canada",
            "ground_truth_citations": ["maple_leaf_flag"],
            "metadata": {
                "final_answer": "The answer is Canada.",
                "cited_chunk_ids": ["maple_leaf_flag"],
                "steps_count": 3,
                "invalid_action_count": 0
            }
        }
        reward = self.loop.run_until_complete(custom_rm(self.args, sample))
        
        # Base accuracy = 1.0 (contains) + 0.25 (token_f1 is 0.5 * 0.5) + 0.50 (citation_f1 is 1.0 * 0.5) = 1.75
        # Deductions: 3 steps * 0.01 = 0.03
        # Expected: 1.75 - 0.03 = 1.72
        self.assertAlmostEqual(reward, 1.72, places=4)

    def test_custom_reward_penalties(self):
        sample = {
            "ground_truth_answer": "Canada",
            "ground_truth_citations": ["maple_leaf_flag"],
            "metadata": {
                "final_answer": "Wrong Answer",
                "cited_chunk_ids": [],
                "steps_count": 5,
                "invalid_action_count": 2
            }
        }
        reward = self.loop.run_until_complete(custom_rm(self.args, sample))
        
        # Base accuracy = 0.0. Citation accuracy = 0.0.
        # Deductions: 5 steps * 0.01 = 0.05. 2 invalid * 0.05 = 0.10.
        # Expected reward: 0.0 - 0.05 - 0.10 = -0.15
        self.assertAlmostEqual(reward, -0.15, places=4)

if __name__ == "__main__":
    unittest.main()
