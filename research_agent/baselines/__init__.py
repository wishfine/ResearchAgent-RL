"""Inference-time policies for the canonical ``research_agent`` environment."""

from .llm_actor import LLMActor, ActorTurn

__all__ = ["LLMActor", "ActorTurn"]
