from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Chunk:
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
    doc_id: str
    title: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)


@dataclass
class CandidateChunk:
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
    chunk_id: str
    summary: str
    key_claims: List[str] = field(default_factory=list)
    evidence_strength: float = 1.0
    key_passages: List[str] = field(default_factory=list)

    def has_evidence(self) -> bool:
        return self.evidence_strength > 0.3

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "summary": self.summary,
            "key_claims": self.key_claims,
            "evidence_strength": self.evidence_strength,
            "key_passages": self.key_passages,
        }
