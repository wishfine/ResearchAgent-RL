"""
src/baselines/rule_based.py
职责: 最小 rule-based baseline 策略
设计: 基于规则的启发式决策；用于演示 environment 运行逻辑；后续可替换为 learned policy
"""
from __future__ import annotations
from typing import Optional
from ..schema.observation import Observation
from ..schema.action import Action


class RuleBasedPolicy:
    """
    基于规则的启发式策略。

    决策逻辑（优先级从高到低）：
    1. remaining_steps <= 1 → 强制 ANSWER
    2. no_progress >= 2 → 如果有 citation 则 ANSWER，否则 SEARCH 新 query
    3. candidate_chunks 为空 → SEARCH
    4. 有候选但 READ 不足 → READ
    5. READ 够但 CITE 不足 → CITE
    6. CITE 够且 remaining_steps 不多 → ANSWER
    7. 默认 → RERANK 或 ANSWER
    """

    def decide(self, obs: Observation) -> Action:
        ctx = obs.get_decision_context()

        # === 优先级 1：强制结束 ===
        if ctx["remaining_steps"] <= 1:
            return self._make_answer(obs)

        # === 优先级 2：无进展处理 ===
        if ctx["no_progress"] >= 2:
            if ctx["n_cited"] >= 2:
                return self._make_answer(obs)
            else:
                return self._make_search(obs, explore=True)

        # === 优先级 3：初始检索 ===
        if ctx["n_candidates"] == 0:
            return self._make_search(obs)

        # === 优先级 4：读取候选 chunk ===
        min_read = min(ctx["n_candidates"], 3)
        if ctx["n_read"] < min_read:
            unread_ids = self._get_unread_candidate_ids(obs)
            if unread_ids:
                return Action.read(
                    chunk_ids=unread_ids[:3],
                    read_goal="extract key claims, methods, and experimental results",
                )

        # === 优先级 5：引用证据 ===
        if ctx["n_cited"] < min(ctx["n_read"], 3):
            citeable = self._get_citeable_chunk_ids(obs)
            if citeable:
                claims = self._infer_claims_from_summaries(obs)
                return Action.cite(
                    chunk_ids=citeable[:3],
                    claims=claims[:3],
                )

        # === 优先级 6：可以作答 ===
        if ctx["n_cited"] >= 2 and ctx["remaining_steps"] <= 5:
            return self._make_answer(obs)

        # === 默认：RERANK 找更好的候选 ===
        top_candidates = self._get_topk_candidate_ids(obs, k=5)
        if top_candidates:
            return Action.rerank(
                query=obs.user_query,
                candidate_chunk_ids=top_candidates,
                topk=3,
                rerank_goal="prioritize chunks with strongest evidence for the query",
            )

        # fallback：直接作答
        return self._make_answer(obs)

    def _make_search(self, obs: Observation, explore: bool = False) -> Action:
        """构造 SEARCH action。"""
        if explore:
            query = obs.current_evidence_gap or obs.user_query
        else:
            query = obs.user_query
        return Action.search(
            query=query,
            topk=10,
            target_gap=obs.current_evidence_gap,
            query_rationale=f"Initial search for: {query[:50]}",
        )

    def _make_answer(self, obs: Observation) -> Action:
        """构造 ANSWER action。"""
        answer_parts = [f"Query: {obs.user_query}\n\n"]

        if obs.read_summaries:
            answer_parts.append("Evidence from cited sources:\n")
            for summary in obs.read_summaries[:5]:
                answer_parts.append(f"- {summary.summary}\n")
                if summary.key_claims:
                    for claim in summary.key_claims[:2]:
                        answer_parts.append(f"  • {claim}\n")

        answer_text = "".join(answer_parts)
        return Action.answer(
            answer_text=answer_text,
            cited_chunk_ids=obs.cited_chunks.copy(),
        )

    def _get_unread_candidate_ids(self, obs: Observation) -> list:
        read_ids = {s.chunk_id for s in obs.read_summaries}
        return [c.chunk_id for c in obs.candidate_chunks if c.chunk_id not in read_ids]

    def _get_citeable_chunk_ids(self, obs: Observation) -> list:
        """获取可以 cite 的 chunk ids（读过且有证据的）。"""
        citeable = []
        for summary in obs.read_summaries:
            if summary.has_evidence():
                citeable.append(summary.chunk_id)
        cited = set(obs.cited_chunks)
        return [cid for cid in citeable if cid not in cited]

    def _infer_claims_from_summaries(self, obs: Observation) -> list:
        """从 read_summaries 推断 claims。"""
        claims = []
        for summary in obs.read_summaries:
            for claim in summary.key_claims:
                claims.append(claim)
        return claims[:5]

    def _get_topk_candidate_ids(self, obs: Observation, k: int = 5) -> list:
        """获取 top-k candidate chunk ids。"""
        sorted_chunks = sorted(obs.candidate_chunks, key=lambda c: c.rank)
        return [c.chunk_id for c in sorted_chunks[:k]]
