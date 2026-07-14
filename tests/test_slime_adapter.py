from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock
import asyncio
import json

from research_agent.adapters.slime.custom_generator import custom_generate
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
        
        # Slime Sample fields
        self.prompt = ""
        self.response = ""
        self.tokens = []
        self.response_length = 0
        self.loss_mask = []
        self.metadata = {}


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
        # We mock a citation-grounded sequence of four HTTP responses:
        # 1. Search action response
        # 2. Read action response
        # 3. Cite action response
        # 4. Answer action response
        
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
            "choices": [{"text": '<reasoning>cite</reasoning><action>{"tool": "CITE", "params": {"chunk_ids": ["maple_leaf_flag"], "claims": ["Canada has a maple leaf flag."]}}</action>'}]
        }).encode("utf-8")

        mock_resp_4 = MagicMock()
        mock_resp_4.read.return_value = json.dumps({
            "choices": [{"text": '<reasoning>answer</reasoning><action>{"tool": "ANSWER", "params": {"answer_text": "The answer is Canada.", "cited_chunk_ids": ["maple_leaf_flag"]}}</action>'}]
        }).encode("utf-8")

        mock_enter_1 = MagicMock()
        mock_enter_1.__enter__.return_value = mock_resp_1
        
        mock_enter_2 = MagicMock()
        mock_enter_2.__enter__.return_value = mock_resp_2
        
        mock_enter_3 = MagicMock()
        mock_enter_3.__enter__.return_value = mock_resp_3
        mock_enter_4 = MagicMock()
        mock_enter_4.__enter__.return_value = mock_resp_4

        mock_urlopen.side_effect = [mock_enter_1, mock_enter_2, mock_enter_3, mock_enter_4]

        # Execute custom_generate asynchronously
        result = self.loop.run_until_complete(
            custom_generate(self.args, self.input_sample, self.sampling_params)
        )

        # Assert the returned object is the modified input sample
        self.assertIs(result, self.input_sample)
        
        # 1. Assert type(out).__name__ matches or is duck-type Sample
        self.assertTrue(type(result).__name__ in {"Sample", "SlimeMockInputSample"})
        
        # 2. Assert len(out.tokens) == len(out.loss_mask)
        self.assertEqual(len(result.tokens), len(result.loss_mask))
        
        # 3. Assert out.response_length == sum(out.loss_mask)
        self.assertEqual(result.response_length, sum(result.loss_mask))
        self.assertTrue(result.response_length > 0)
        self.assertTrue(result.response_length <= len(result.tokens))
        
        # 4. Assert out.metadata["final_answer"] is not empty
        self.assertTrue(bool(result.metadata["final_answer"]))
        self.assertEqual(result.metadata["final_answer"], "The answer is Canada.")
        
        # 5. Assert out.metadata["done_reason"] is answer_submitted
        self.assertEqual(result.metadata["done_reason"], "answer_submitted")
        self.assertTrue(result.metadata["done_reason"] in {"answer_submitted", "max_steps"})

        # Verify other rollout metrics
        self.assertEqual(result.metadata["steps_count"], 4)
        self.assertEqual(result.metadata["invalid_action_count"], 0)
        self.assertEqual(result.metadata["cited_chunk_ids"], ["maple_leaf_flag"])
        
        # Verify that prompt is exactly system + user
        self.assertTrue(result.prompt.startswith("<|im_start|>system"))
        self.assertTrue(result.prompt.endswith("features a red maple leaf?<|im_end|>\n"))

    def test_custom_generator_mock_mode_deterministic(self):
        # Configure actor URL to 'mock' to activate direct deterministic generation
        self.args.actor_model_url = "mock"
        
        # Run generate without patching urllib
        result = self.loop.run_until_complete(
            custom_generate(self.args, self.input_sample, self.sampling_params)
        )

        self.assertIs(result, self.input_sample)
        
        # Verify all core assertions are met in mock mode
        self.assertTrue(type(result).__name__ in {"Sample", "SlimeMockInputSample"})
        self.assertEqual(len(result.tokens), len(result.loss_mask))
        self.assertEqual(result.response_length, sum(result.loss_mask))
        self.assertTrue(result.response_length > 0)
        self.assertTrue(result.response_length <= len(result.tokens))
        self.assertTrue(bool(result.metadata["final_answer"]))
        self.assertEqual(result.metadata["done_reason"], "answer_submitted")
        self.assertTrue(result.metadata["done_reason"] in {"answer_submitted", "max_steps"})

        self.assertEqual(result.metadata["steps_count"], 4)
        self.assertEqual(result.metadata["invalid_action_count"], 0)
        self.assertEqual(result.metadata["cited_chunk_ids"], ["maple_leaf_flag"])

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

    def test_custom_reward_reads_verifier_fields_from_metadata(self):
        """The Vime dataset converter deliberately keeps task fields in metadata."""
        sample = {
            "metadata": {
                "ground_truth_answer": "Canada",
                "ground_truth_citations": ["maple_leaf_flag"],
                "final_answer": "Canada",
                "cited_chunk_ids": ["maple_leaf_flag"],
                "steps_count": 1,
                "invalid_action_count": 0,
            }
        }
        reward = self.loop.run_until_complete(custom_rm(self.args, sample))
        self.assertAlmostEqual(reward, 1.99, places=4)

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
