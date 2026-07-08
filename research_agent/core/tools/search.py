from __future__ import annotations
from .base import BaseTool, ToolResult
from research_agent.core.env.state import EnvState

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
