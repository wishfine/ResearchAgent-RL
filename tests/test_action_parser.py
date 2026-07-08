from __future__ import annotations
import unittest
from research_agent.core.schema.parser import ActionParser
from research_agent.core.schema.action import Action

class TestActionParser(unittest.TestCase):
    def test_parse_standard_json(self):
        text = """
        <reasoning>I need to search for Canada flag.</reasoning>
        <action>
        {
            "tool": "SEARCH",
            "params": {"query": "Canada flag", "topk": 5},
            "intent": "Search for Canada's flag description"
        }
        </action>
        """
        action = ActionParser.parse(text)
        self.assertEqual(action.tool, "SEARCH")
        self.assertEqual(action.params["query"], "Canada flag")
        self.assertEqual(action.params["topk"], 5)
        self.assertEqual(action.reasoning, "I need to search for Canada flag.")

    def test_parse_markdown_json(self):
        text = """
        Thinking process here.
        <action>
        ```json
        {
            "tool": "READ",
            "params": {"chunk_ids": ["doc1_c1", "doc1_c2"]}
        }
        ```
        </action>
        """
        action = ActionParser.parse(text)
        self.assertEqual(action.tool, "READ")
        self.assertEqual(action.params["chunk_ids"], ["doc1_c1", "doc1_c2"])
        self.assertEqual(action.reasoning, "Thinking process here.")

    def test_parse_fallback_kv(self):
        text = """
        I will answer now.
        <action>
        tool: ANSWER
        answer_text: Canada is the country.
        cited_chunk_ids: [doc1_c1]
        </action>
        """
        action = ActionParser.parse(text)
        self.assertEqual(action.tool, "ANSWER")
        self.assertEqual(action.params["answer_text"], "Canada is the country.")
        self.assertEqual(action.params["cited_chunk_ids"], ["doc1_c1"])

    def test_parse_invalid(self):
        text = "This is some garbage output with no structured action."
        action = ActionParser.parse(text)
        self.assertEqual(action.tool, "INVALID")

if __name__ == "__main__":
    unittest.main()
