from __future__ import annotations
import unittest
from research_agent.core.env.rollout_collector import ConversationCollector, RolloutSegment
from research_agent.core.env.mask_builder import build_loss_mask, build_labels, MockTokenizer
from research_agent.core.schema.mock_policy import MockLLMPolicy

class TestMockRolloutAndMasking(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()

    def test_segment_to_chatml(self):
        seg = RolloutSegment(role="user", text="hello", is_trainable=False)
        self.assertEqual(seg.to_chatml(), "<|im_start|>user\nhello<|im_end|>\n")

    def test_conversation_collector(self):
        collector = ConversationCollector(system_prompt="sys")
        collector.add_user_message("query")
        collector.add_assistant_response("thought")
        
        full_text = collector.get_full_text()
        expected = (
            "<|im_start|>system\nsys<|im_end|>\n"
            "<|im_start|>user\nQuestion: query<|im_end|>\n"
            "<|im_start|>assistant\nthought<|im_end|>\n"
        )
        self.assertEqual(full_text, expected)
        
        # Test prompt for next generation includes assistant start wrapper
        self.assertEqual(
            collector.get_prompt_for_generation(),
            expected + "<|im_start|>assistant\n"
        )

    def test_default_prompt_declares_exact_schema_for_every_tool(self):
        prompt = ConversationCollector().segments[0].text

        self.assertIn('<action>{"tool":"SEARCH","intent":"...","params":{"query":"...","topk":3}}</action>', prompt)
        self.assertIn('<action>{"tool":"READ","intent":"...","params":{"chunk_ids":["..."]}}</action>', prompt)
        self.assertIn('<action>{"tool":"RERANK","intent":"...","params":{"query":"...","candidate_chunk_ids":["..."],"topk":3}}</action>', prompt)
        self.assertIn('<action>{"tool":"CITE","intent":"...","params":{"chunk_ids":["..."],"claims":["..."]}}</action>', prompt)
        self.assertIn('<action>{"tool":"ANSWER","intent":"...","params":{"answer_text":"...","cited_chunk_ids":["..."]}}</action>', prompt)
        self.assertIn("SEARCH -> READ -> CITE -> ANSWER", prompt)
        self.assertIn("Never call ANSWER with an empty cited_chunk_ids list", prompt)
        self.assertIn("never put action fields beside params", prompt)
        self.assertIn("After a successful SEARCH with candidates, READ a returned chunk next", prompt)

    def test_build_loss_mask_boundaries(self):
        segments = [
            RolloutSegment(role="user", text="Hello", is_trainable=False),
            RolloutSegment(role="assistant", text="Hi", is_trainable=True),
        ]
        
        input_ids, loss_mask = build_loss_mask(segments, self.tokenizer)
        
        # Reconstruct text from input_ids
        reconstructed = self.tokenizer.decode(input_ids)
        expected_full_text = (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi<|im_end|>\n"
        )
        self.assertEqual(reconstructed, expected_full_text)
        
        # Check that loss mask is exactly 1 for the assistant inner content ("Hi") and 0 elsewhere
        # Expected sequence:
        # User turn (33 chars): "<|im_start|>user\nHello<|im_end|>\n" -> 33 zeros
        # Assistant prefix (22 chars): "<|im_start|>assistant\n" -> 22 zeros
        # Assistant inner (2 chars): "Hi" -> 2 ones
        # Assistant suffix (11 chars): "<|im_end|>\n" -> 11 zeros
        
        expected_mask = [0] * 33 + [0] * 22 + [1] * 2 + [0] * 11
        self.assertEqual(loss_mask, expected_mask)

        # Test labels building
        labels = build_labels(input_ids, loss_mask, ignore_index=-100)
        # All masked items should be -100
        self.assertEqual(labels[:33 + 22], [-100] * (33 + 22))
        # Trainable items should be equal to input_ids
        self.assertEqual(labels[33 + 22: 33 + 22 + 2], input_ids[33 + 22: 33 + 22 + 2])
        self.assertEqual(labels[33 + 22 + 2:], [-100] * 11)

if __name__ == "__main__":
    unittest.main()
