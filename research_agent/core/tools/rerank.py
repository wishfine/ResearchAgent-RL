from __future__ import annotations

from research_agent.core.env.state import EnvState
from .base import BaseTool, ToolResult


class RerankTool(BaseTool):
    name = "RERANK"
    description = "Rerank retrieved candidates using query-term overlap"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        candidate_ids = params["candidate_chunk_ids"]
        available = {candidate.chunk_id: candidate for candidate in state.candidate_chunks}
        unknown = [chunk_id for chunk_id in candidate_ids if chunk_id not in available]
        if unknown:
            raise ValueError(f"RERANK candidates were not retrieved: {unknown}")

        query_terms = set(params["query"].lower().split())
        ranked = sorted(
            (available[chunk_id] for chunk_id in candidate_ids),
            key=lambda candidate: (
                sum(term in candidate.snippet.lower() for term in query_terms),
                candidate.score,
            ),
            reverse=True,
        )
        selected = ranked[: params["topk"]]
        state.rerank_candidates([candidate.chunk_id for candidate in selected])
        return ToolResult(
            success=True,
            data={"candidates": [candidate.to_dict() for candidate in selected]},
            stats={"n_reranked": len(ranked), "n_selected": len(selected)},
        )
