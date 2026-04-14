"""
src/schema/document.py
职责: 定义文档和 chunk 相关的数据模型
设计: Chunk 是检索和引用的原子单位; Document 聚合多个 Chunk; CandidateChunk 携带检索 score; ReadSummary 承载阅读后提取的 claim
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """检索和引用的原子单位。"""
    chunk_id: str
    doc_id: str
    content: str
    char_start: int
    char_end: int
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""

    def __repr__(self) -> str:
        return f"Chunk({self.chunk_id}: {self.content[:40]}...)"


@dataclass
class Document:
    """聚合多个 Chunk 的文档单位。"""
    doc_id: str
    title: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)


@dataclass
class CandidateChunk:
    """
    来自 SEARCH/RERANK 的候选 chunk。
    携带检索 score 和来源 query，便于分析检索质量。
    """
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    query: str
    title: str = ""
    snippet: str = ""

    def to_chunk_ref(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "rank": self.rank}

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "rank": self.rank,
            "query": self.query,
            "title": self.title,
            "snippet": self.snippet,
        }


@dataclass
class ReadSummary:
    """
    READ tool 执行后的摘要结果。
    承载从 chunk 中提取的关键 claim 和证据强度。
    """
    chunk_id: str
    summary: str
    key_claims: List[str] = field(default_factory=list)
    evidence_strength: float = 0.0
    key_passages: List[str] = field(default_factory=list)

    def has_evidence(self) -> bool:
        return self.evidence_strength > 0.3 and len(self.key_claims) > 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "summary": self.summary,
            "key_claims": self.key_claims,
            "evidence_strength": self.evidence_strength,
            "key_passages": self.key_passages,
        }
