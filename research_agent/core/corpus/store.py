from __future__ import annotations
import os
import json
import re
import sqlite3
import sys
from typing import List, Dict, Optional, Set
from research_agent.core.schema.document import Chunk, Document, CandidateChunk

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

class CorpusStore:
    def __init__(self, corpus_dir: str = "data/corpus"):
        self.corpus_dir = corpus_dir
        self.chunks: Dict[str, Chunk] = {}
        self.docs: Dict[str, Document] = {}
        self._doc_ids: Set[str] = set()
        self._chunk_ids: Set[str] = set()
        
        # BM25 specific
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_chunk_ids: List[str] = []
        self._sqlite: Optional[sqlite3.Connection] = None
        self._sqlite_count = 0

    def close(self) -> None:
        if getattr(self, "_sqlite", None) is not None:
            self._sqlite.close()
            self._sqlite = None
            self._sqlite_count = 0

    def __del__(self) -> None:
        self.close()

    def load(self) -> None:
        """Open a pooled SQLite index or load the legacy JSON corpus."""
        if not os.path.exists(self.corpus_dir):
            return

        # Pooled multi-hop corpora are indexed on disk. Loading every passage
        # into Python and scoring the entire split for each SEARCH is too slow
        # for RL rollouts, and makes every Ray worker duplicate the index.
        index_path = os.path.join(self.corpus_dir, "corpus.sqlite")
        if os.path.isfile(index_path):
            self._sqlite = sqlite3.connect(
                f"file:{index_path}?mode=ro", uri=True, check_same_thread=False
            )
            self._sqlite.row_factory = sqlite3.Row
            ready = self._sqlite.execute(
                "SELECT value FROM index_metadata WHERE key = 'complete'"
            ).fetchone()
            if ready is None or ready[0] != "1":
                raise ValueError(f"Corpus index is incomplete: {index_path}")
            self._sqlite_count = self._sqlite.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return

        for filename in os.listdir(self.corpus_dir):
            if not filename.endswith(".json") or filename.startswith("."):
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

        # Initialize BM25 if library is present and chunks loaded
        if HAS_BM25 and self.chunks:
            self.bm25_chunk_ids = list(self.chunks.keys())
            tokenized_corpus = [self.chunks[cid].content.lower().split() for cid in self.bm25_chunk_ids]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print("[CorpusStore] rank_bm25 not installed, fallback to simple_search", file=sys.stderr)

    def _add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self._chunk_ids.add(chunk.chunk_id)
        self._doc_ids.add(chunk.doc_id)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        if self._sqlite is not None:
            row = self._sqlite.execute(
                "SELECT chunk_id, doc_id, title, content FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                return None
            return Chunk(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                title=row["title"], content=row["content"],
                char_start=0, char_end=len(row["content"]),
            )
        return self.chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        chunks = [self.get_chunk(cid) for cid in chunk_ids]
        return [chunk for chunk in chunks if chunk is not None]

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        # FTS5's unicode61 tokenizer splits on punctuation. Quote each term
        # before composing MATCH so a model-generated query is data, not SQL.
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
            "does", "for", "from", "how", "in", "is", "it", "of", "on",
            "or", "the", "to", "was", "were", "what", "when", "where",
            "which", "who", "why", "with",
        }
        terms = [term.casefold() for term in re.findall(r"[^\W_]+", query, re.UNICODE)]
        terms = list(dict.fromkeys(term for term in terms if term not in stopwords))
        return (terms or list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold()))))[:24]

    def _search_sqlite(
        self, query: str, topk: int, doc_ids: Optional[List[str]]
    ) -> List[CandidateChunk]:
        if topk <= 0 or doc_ids == []:
            return []
        terms = self._query_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)
        scope = ""
        arguments: list = [match]
        if doc_ids is not None:
            scope = f" AND chunks.doc_id IN ({','.join('?' for _ in doc_ids)})"
            arguments.extend(doc_ids)
        arguments.append(topk)
        rows = self._sqlite.execute(
            "SELECT chunks.chunk_id, chunks.doc_id, chunks.title, chunks.content, "
            "bm25(chunk_fts, 5.0, 1.0) AS score "
            "FROM chunk_fts JOIN chunks ON chunks.rowid = chunk_fts.rowid "
            f"WHERE chunk_fts MATCH ?{scope} "
            "ORDER BY score ASC, chunks.chunk_id ASC LIMIT ?",
            arguments,
        ).fetchall()
        return [
            CandidateChunk(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                score=-float(row["score"]), rank=rank, query=query,
                title=row["title"], snippet=row["content"][:200],
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def search_simple(
        self, query: str, topk: int = 10, doc_ids: Optional[List[str]] = None
    ) -> List[CandidateChunk]:
        """Simple substring frequency count search."""
        if self._sqlite is not None:
            return self._search_sqlite(query, topk, doc_ids)
        query_terms = query.lower().split()
        scores: Dict[str, float] = {}

        chunk_ids_to_search = set(self._chunk_ids)
        if doc_ids is not None:
            # HotpotQA retrieval is normally scoped to a single task document.
            # Use the document index when it is available instead of scanning
            # every split chunk for every query.
            indexed_docs = [self.docs.get(doc_id) for doc_id in doc_ids]
            if all(document is not None for document in indexed_docs):
                chunk_ids_to_search = {
                    chunk.chunk_id
                    for document in indexed_docs
                    for chunk in document.chunks
                }
            else:
                chunk_ids_to_search = {
                    cid for cid, chunk in self.chunks.items()
                    if chunk.doc_id in doc_ids
                }

        for chunk_id in chunk_ids_to_search:
            chunk = self.chunks[chunk_id]
            content_lower = chunk.content.lower()
            score = sum(content_lower.count(term) for term in query_terms)
            if score > 0:
                scores[chunk_id] = float(score)

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

    def search(
        self, query: str, topk: int = 10, doc_ids: Optional[List[str]] = None
    ) -> List[CandidateChunk]:
        """Performs search. Prefers BM25 if available, otherwise falls back to simple search."""
        if self._sqlite is not None:
            return self._search_sqlite(query, topk, doc_ids)
        if HAS_BM25 and self.bm25 is not None:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            
            # Map index to chunk_id
            candidates_data = []
            for idx, score in enumerate(scores):
                chunk_id = self.bm25_chunk_ids[idx]
                chunk = self.chunks[chunk_id]
                
                # Filter by doc_ids if specified
                if doc_ids is not None and chunk.doc_id not in doc_ids:
                    continue
                    
                # rank_bm25 may assign zero/negative IDF to common matching
                # terms. Score sign is not a test for lexical relevance.
                # This legacy-JSON fix does not change SQLite FTS5 ranking.
                if set(query_tokens) & set(chunk.content.lower().split()):
                    candidates_data.append((chunk_id, float(score)))
                    
            candidates_data.sort(key=lambda x: (-x[1], x[0]))
            
            candidates = []
            for rank, (chunk_id, score) in enumerate(candidates_data[:topk], start=1):
                chunk = self.chunks[chunk_id]
                snippet = chunk.content[:200]
                candidates.append(CandidateChunk(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    score=score,
                    rank=rank,
                    query=query,
                    title=chunk.title,
                    snippet=snippet,
                ))
            return candidates
        else:
            return self.search_simple(query, topk=topk, doc_ids=doc_ids)

    def __len__(self) -> int:
        return self._sqlite_count if self._sqlite is not None else len(self.chunks)

    def __contains__(self, chunk_id: str) -> bool:
        if self._sqlite is not None:
            return self.get_chunk(chunk_id) is not None
        return chunk_id in self._chunk_ids
