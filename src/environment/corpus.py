"""
src/environment/corpus.py
职责: CorpusStore——本地文档库管理
设计: 加载目录中的 chunk JSON 文件；提供 doc_ids/chunk_ids 检索；预留 BM25 检索接口
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import json
import os

from ..schema.document import Chunk, Document, CandidateChunk


class CorpusStore:
    """
    本地文档库管理器。
    """

    def __init__(self, corpus_dir: str = "data/corpus"):
        self.corpus_dir = corpus_dir
        self.chunks: Dict[str, Chunk] = {}
        self.docs: Dict[str, Document] = {}
        self._doc_ids: Set[str] = set()
        self._chunk_ids: Set[str] = set()

    def load(self) -> None:
        """加载 corpus_dir 下所有 JSON 文件。"""
        if not os.path.exists(self.corpus_dir):
            return

        for filename in os.listdir(self.corpus_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.corpus_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "chunks" in data:
                doc = Document(
                    doc_id=data.get("doc_id", filename[:-5]),
                    title=data.get("title", ""),
                    metadata=data.get("metadata", {}),
                )
                for chunk_data in data["chunks"]:
                    chunk = Chunk(
                        chunk_id=chunk_data["chunk_id"],
                        doc_id=doc.doc_id,
                        content=chunk_data["content"],
                        char_start=chunk_data.get("char_start", 0),
                        char_end=chunk_data.get("char_end", len(chunk_data["content"])),
                        title=chunk_data.get("title", doc.title),
                        authors=chunk_data.get("authors", []),
                        year=chunk_data.get("year"),
                        venue=chunk_data.get("venue", ""),
                    )
                    self._add_chunk(chunk)
                    doc.add_chunk(chunk)
                self.docs[doc.doc_id] = doc
            elif "chunk_id" in data:
                chunk = Chunk(
                    chunk_id=data["chunk_id"],
                    doc_id=data.get("doc_id", ""),
                    content=data["content"],
                    char_start=data.get("char_start", 0),
                    char_end=data.get("char_end", len(data["content"])),
                    title=data.get("title", ""),
                    authors=data.get("authors", []),
                    year=data.get("year"),
                    venue=data.get("venue", ""),
                )
                self._add_chunk(chunk)

    def _add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self._chunk_ids.add(chunk.chunk_id)
        self._doc_ids.add(chunk.doc_id)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        """批量获取 chunks，不存在的忽略。"""
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def get_doc_ids(self) -> List[str]:
        return list(self._doc_ids)

    def search_simple(
        self, query: str, topk: int = 10, doc_ids: Optional[List[str]] = None
    ) -> List[CandidateChunk]:
        """
        简单文本检索（当前版本）。
        """
        query_terms = query.lower().split()
        scores: Dict[str, float] = {}

        chunk_ids_to_search = set(self._chunk_ids)
        if doc_ids is not None:
            chunk_ids_to_search = {
                cid for cid, chunk in self.chunks.items()
                if chunk.doc_id in doc_ids
            }

        for chunk_id in chunk_ids_to_search:
            chunk = self.chunks[chunk_id]
            content_lower = chunk.content.lower()
            score = sum(content_lower.count(term) for term in query_terms)
            if score > 0:
                scores[chunk_id] = score

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        candidates = []
        for rank, chunk_id in enumerate(sorted_ids[:topk], start=1):
            chunk = self.chunks[chunk_id]
            snippet = chunk.content[:200]
            candidates.append(CandidateChunk(
                chunk_id=chunk_id,
                doc_id=chunk.doc_id,
                score=scores[chunk_id],
                rank=rank,
                query=query,
                title=chunk.title,
                snippet=snippet,
            ))
        return candidates

    def __len__(self) -> int:
        return len(self.chunks)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunk_ids
