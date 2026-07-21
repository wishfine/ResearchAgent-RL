from __future__ import annotations
from .base import BaseTool, ToolResult
from research_agent.core.env.state import EnvState
from research_agent.core.schema.document import CandidateChunk

class SearchTool(BaseTool):
    name = "SEARCH"
    description = "Search corpus for chunks relevant to query"

    def execute(self, params: dict, state: EnvState) -> ToolResult:
        query = params["query"]
        topk = params.get("topk", 10)

        corpus = state.get_corpus()
        if corpus is None:
            return ToolResult(success=False, error="Corpus not available")

        # Optionally restrict docs if specified in the task sample
        task = state.task
        doc_ids = task.reference_docs if task.reference_docs else None

        results = corpus.search(query, topk=topk, doc_ids=doc_ids)

        # HotpotQA distractor tasks intentionally scope retrieval to a small
        # per-task candidate set.  When the caller asks for at least that many
        # results, return the complete set, including lexical zero-score
        # passages.  Otherwise a supporting paragraph can be absent from a
        # SEARCH observation while still being a valid corpus chunk, which
        # makes a SEARCH -> READ trajectory impossible to reproduce.
        if doc_ids:
            scoped_chunks = [
                chunk
                for chunk in corpus.chunks.values()
                if chunk.doc_id in set(doc_ids)
            ]
            if len(scoped_chunks) <= topk:
                seen_chunk_ids = {candidate.chunk_id for candidate in results}
                for chunk in sorted(scoped_chunks, key=lambda item: item.chunk_id):
                    if chunk.chunk_id in seen_chunk_ids:
                        continue
                    results.append(CandidateChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        score=0.0,
                        rank=len(results) + 1,
                        query=query,
                        title=chunk.title,
                        snippet=chunk.content[:200],
                    ))

        new_candidates = state.add_candidates(results)
        n_new = len(new_candidates)

        if results:
            state.search_history.append({
                "query": query,
                "topk": topk,
                "n_results": len(results),
                "n_new": n_new,
            })
        else:
            state.add_failed_search(query)
            state.search_history.append({
                "query": query,
                "topk": topk,
                "n_results": 0,
                "failed": True,
            })

        return ToolResult(
            success=True,
            data={
                "candidates": [c.to_dict() for c in results],
                "n_candidates": len(results),
                "n_new": n_new,
                "query": query,
            },
            stats={
                "n_candidates": len(results),
                "n_new": n_new,
            }
        )

    def validate_params(self, params: dict) -> tuple[bool, str]:
        if "query" not in params:
            return False, "SEARCH requires 'query' param"
        return True, ""
