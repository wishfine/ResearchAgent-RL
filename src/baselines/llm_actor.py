"""Compatibility import for the canonical LLM baseline implementation.

New rollout code uses ``research_agent.baselines`` because it matches the
currently exercised environment package.  This module keeps the milestone's
``src.baselines.llm_actor`` import path available without duplicating policy
logic across the two historical package trees.
"""

from research_agent.baselines.llm_actor import ActorTurn, LLMActor

__all__ = ["LLMActor", "ActorTurn"]
