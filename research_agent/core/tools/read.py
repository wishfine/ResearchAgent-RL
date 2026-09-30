from __future__ import annotations
from .base import BaseTool, ToolResult
from research_agent.core.env.state import EnvState
from research_agent.core.schema.document import ReadSummary

class ReadTool(BaseTool):
    name = "READ"
    description = "Read chunks and extract key passages"

    def __init__(self, mode: str = "head300"):
        if mode not in {"head300", "full"}:
            raise ValueError("READ mode must be head300 or full")
        self.mode = mode

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        valid, error = self.validate_params(params)
        if not valid:
            raise ValueError(error)
        chunk_ids = params["chunk_ids"]

        if not chunk_ids:
            return ToolResult(success=False, error="chunk_ids is empty")

        corpus = state.get_corpus()
        if corpus is None:
            return ToolResult(success=False, error="Corpus not available")

        # READ is deliberately capability-scoped: a model may only open a
        # chunk that a preceding SEARCH exposed for this task.  Merely
        # guessing a corpus-wide chunk ID must not bypass retrieval or leak
        # evidence from another task split.
        candidate_ids = {candidate.chunk_id for candidate in state.candidate_chunks}
        not_returned_by_search = [chunk_id for chunk_id in chunk_ids if chunk_id not in candidate_ids]
        if not_returned_by_search:
            raise ValueError(
                "READ requires chunk_ids returned by SEARCH first: "
                f"{not_returned_by_search[:5]}"
            )

        # SEARCH normally already enforces this restriction.  Keep a second
        # check here so future search implementations cannot accidentally
        # widen the task's document boundary.
        allowed_doc_ids = set(state.task.reference_docs)
        if allowed_doc_ids:
            out_of_scope = [
                chunk_id
                for chunk_id in chunk_ids
                if (chunk := corpus.get_chunk(chunk_id)) is None
                or chunk.doc_id not in allowed_doc_ids
            ]
            if out_of_scope:
                raise ValueError(
                    "READ requires chunk_ids from this task's reference_docs: "
                    f"{out_of_scope[:5]}"
                )

        chunks = corpus.get_chunks(chunk_ids)
        if not chunks:
            return ToolResult(success=False, error=f"No chunks found for IDs: {chunk_ids}")

        summaries = []
        n_actually_new = 0
        n_truncated = 0

        for chunk in chunks:
            # Preserve legacy observations unless full-paragraph mode is explicit.
            truncated = self.mode == "head300" and len(chunk.content) > 300
            n_truncated += int(truncated)
            truncated_text = chunk.content[:300] + "..." if truncated else chunk.content
            
            summary = ReadSummary(
                chunk_id=chunk.chunk_id,
                summary=truncated_text,
                key_claims=[truncated_text],
                evidence_strength=1.0,
                key_passages=[chunk.content if self.mode == "full" else chunk.content[:300]]
            )
            
            is_new = state.add_read_summary(summary)
            if is_new:
                n_actually_new += 1
            elif self.mode == "full":
                # A full re-read must refresh the evidence stored in state too.
                state.read_summaries = [summary if old.chunk_id == chunk.chunk_id else old
                                        for old in state.read_summaries]

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
                "n_truncated": n_truncated,
                "read_mode": self.mode,
            }
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "chunk_ids" not in params:
            return False, "READ requires 'chunk_ids' param"
        if not isinstance(params["chunk_ids"], list):
            return False, "READ 'chunk_ids' must be list"
        if any(not isinstance(cid, str) or not cid for cid in params["chunk_ids"]):
            return False, "READ chunk IDs must be non-empty strings"
        return True, ""
