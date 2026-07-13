from __future__ import annotations
from .base import BaseTool, ToolResult
from research_agent.core.env.state import EnvState

class AnswerTool(BaseTool):
    name = "ANSWER"
    description = "Submit final answer with citations"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        answer_text = params["answer_text"]
        cited_chunk_ids = params.get("cited_chunk_ids", [])

        if not answer_text or len(answer_text.strip()) == 0:
            return ToolResult(success=False, error="answer_text cannot be empty")

        # Validate that cited chunk IDs exist in the corpus (if corpus is available)
        corpus = state.get_corpus()
        invalid_ids = []
        if corpus:
            invalid_ids = [cid for cid in cited_chunk_ids if cid not in corpus]

        if invalid_ids:
            return ToolResult(
                success=False,
                error=f"Invalid chunk IDs cited: {invalid_ids[:5]}",
            )

        uncited_ids = [cid for cid in cited_chunk_ids if cid not in state.cited_chunks]
        if uncited_ids:
            raise ValueError(
                "ANSWER requires cited_chunk_ids to be submitted through CITE first: "
                f"{uncited_ids[:5]}"
            )

        # Record answer and update cited_chunks
        state.final_answer = answer_text
        state.cite_chunks(cited_chunk_ids)

        return ToolResult(
            success=True,
            data={
                "answer_text": answer_text,
                "cited_chunk_ids": cited_chunk_ids,
                "n_citations": len(cited_chunk_ids),
            },
            stats={"answer_length": len(answer_text)}
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "answer_text" not in params:
            return False, "ANSWER requires 'answer_text' param"
        return True, ""
