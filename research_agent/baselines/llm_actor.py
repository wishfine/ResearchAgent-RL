from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Type

from research_agent.core.env.llm_client import LLMClient
from research_agent.core.schema.action import Action
from research_agent.core.schema.parser import ActionParser


@dataclass(frozen=True)
class ActorTurn:
    """One model response together with the parsed environment action."""

    raw_content: str
    action: Action
    reasoning: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int


class LLMActor:
    """Thin policy adapter from an OpenAI-compatible model endpoint to ``Action``."""

    def __init__(self, client: LLMClient, parser: Type[ActionParser] = ActionParser):
        self.client = client
        self.parser = parser

    def decide(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        stop_tokens: Optional[list[str]] = None,
    ) -> ActorTurn:
        started = perf_counter()
        response = self.client.generate_response(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
        )
        latency_sec = perf_counter() - started

        # The action parser only receives visible model content.  vLLM's
        # ``reasoning`` field stays metadata and never contaminates action JSON.
        action = self.parser.parse(response.content)
        reasoning = response.reasoning or action.reasoning or ""
        action.reasoning = reasoning

        return ActorTurn(
            raw_content=response.content,
            action=action,
            reasoning=reasoning,
            latency_sec=latency_sec,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
