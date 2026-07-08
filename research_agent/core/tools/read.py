from __future__ import annotations
from .base import BaseTool, ToolResult
from research_agent.core.env.state import EnvState
from research_agent.core.schema.document import ReadSummary

class ReadTool(BaseTool):
    name = "READ"
    description = "Read chunks and extract key passages"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        chunk_ids = params["chunk_ids"]

        if not chunk_ids:
            return ToolResult(success=False, error="chunk_ids is empty")

        corpus = state.get_corpus()
        if corpus is None:
            return ToolResult(success=False, error="Corpus not available")

        chunks = corpus.get_chunks(chunk_ids)
        if not chunks:
            return ToolResult(success=False, error=f"No chunks found for IDs: {chunk_ids}")

        summaries = []
        n_actually_new = 0

        for chunk in chunks:
            # Deterministic truncation: take first 300 chars
            truncated_text = chunk.content[:300] + ("..." if len(chunk.content) > 300 else "")
            
            summary = ReadSummary(
                chunk_id=chunk.chunk_id,
                summary=truncated_text,
                key_claims=[truncated_text],
                evidence_strength=1.0,
                key_passages=[chunk.content[:300]]
            )
            
            is_new = state.add_read_summary(summary)
            if is_new:
                n_actually_new += 1

            summaries.append({
                "chunk_id": chunk.chunk_id,
                "summary": summary.summary,
                "is_new": is_new,
            })

        return ToolResult(
            success=True,
            data={"summaries": summaries},
            stats={
                "n_read": len(chunks),
                "n_new": n_actually_new,
            }
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "chunk_ids" not in params:
            return False, "READ requires 'chunk_ids' param"
        if not isinstance(params["chunk_ids"], list):
            return False, "READ 'chunk_ids' must be list"
        return True, ""
