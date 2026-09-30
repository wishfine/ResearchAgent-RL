from __future__ import annotations

from research_agent.core.env.state import EnvState
from .base import BaseTool, ToolResult


class CiteTool(BaseTool):
    name = "CITE"
    description = "Attach already-read evidence chunks to explicit claims"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        chunk_ids = params["chunk_ids"]
        claims = params["claims"]
        read_ids = {summary.chunk_id for summary in state.read_summaries}
        unread = [chunk_id for chunk_id in chunk_ids if chunk_id not in read_ids]
        if unread:
            raise ValueError(f"CITE requires chunks to be READ first: {unread}")
        if not claims or not all(isinstance(claim, str) and claim.strip() for claim in claims):
            raise ValueError("CITE requires at least one non-empty claim")

        new_citations = state.cite_chunks(chunk_ids)
        return ToolResult(
            success=True,
            data={"chunk_ids": chunk_ids, "claims": claims, "n_new": len(new_citations)},
            stats={"n_cited": len(chunk_ids), "n_new": len(new_citations)},
        )
