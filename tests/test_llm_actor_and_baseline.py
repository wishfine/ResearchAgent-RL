from __future__ import annotations

import unittest

from research_agent.baselines.llm_actor import LLMActor
from research_agent.core.env.llm_client import LLMResponse
from research_agent.core.schema.action import Action
from research_agent.core.schema.parser import ActionParser


class _FakeClient:
    def __init__(self, response: LLMResponse):
        self.response = response
        self.calls = []

    def generate_response(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


class TestLLMActor(unittest.TestCase):
    def test_uses_content_only_and_preserves_reasoning_metadata(self):
        client = _FakeClient(
            LLMResponse(
                content='<action>{"tool":"SEARCH","params":{"query":"Canada"}}</action>',
                reasoning="The maple leaf identifies Canada.",
                prompt_tokens=11,
                completion_tokens=9,
            )
        )
        actor = LLMActor(client=client, parser=ActionParser)

        turn = actor.decide("prompt", max_tokens=64, temperature=0.0, top_p=1.0)

        self.assertEqual(turn.action.tool, "SEARCH")
        self.assertEqual(turn.reasoning, "The maple leaf identifies Canada.")
        self.assertEqual(turn.prompt_tokens, 11)
        self.assertEqual(turn.completion_tokens, 9)
        self.assertEqual(client.calls[0][1]["top_p"], 1.0)

    def test_parse_failure_is_returned_as_invalid_action(self):
        client = _FakeClient(
            LLMResponse(
                content="SEARCH: Canada",
                reasoning="",
                prompt_tokens=2,
                completion_tokens=2,
            )
        )
        actor = LLMActor(client=client, parser=ActionParser)

        turn = actor.decide("prompt")

        self.assertEqual(turn.action.tool, "INVALID")


if __name__ == "__main__":
    unittest.main()
