"""Optional exact cosine + lexical RRF retrieval. No gold annotations used.

Corpus vectors stay on CPU; only the embedding API uses its dedicated model.
Index windows are internal. SEARCH/READ expose the original paragraph IDs.
"""
from __future__ import annotations

from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np

from research_agent.core.schema.document import CandidateChunk
from research_agent.core.tools.base import ToolInfrastructureError


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


class EmbeddingClient:
    def __init__(self, url, model, query_instruction="", event_log=None, fingerprint=None, tokenizer=None):
        self.url = url.rstrip("/")
        self.model, self.query_instruction = model, query_instruction
        self.event_log = Path(event_log) if event_log else None
        self.fingerprint = fingerprint
        self.tokenizer = tokenizer
        self.input_policy = ("local_hf_token_ids_v1" if tokenizer is not None or
                             (fingerprint and fingerprint.get("pooling") in {"CLS", "LAST"})
                             else "server_text_v1")

    def _event(self, event):
        if self.event_log:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log.open("a") as handle:
                handle.write(json.dumps({"time": time.time(), **event}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def verify_server(self):
        try:
            if self.fingerprint:
                root = Path(self.fingerprint["root"])
                for name, expected in self.fingerprint["files"].items():
                    path = root / name
                    if not path.is_file() or file_hash(path) != expected:
                        raise ValueError(f"Live embedding file fingerprint mismatch: {name}")
            with urllib.request.urlopen(self.url + "/models", timeout=10) as response:
                models = json.load(response).get("data", [])
            matches = [model for model in models if model.get("id") == self.model]
            if len(matches) != 1:
                raise ValueError("Embedding server model ID mismatch")
            if self.fingerprint and matches[0].get("root") != self.fingerprint["root"]:
                raise ValueError("Embedding server root mismatch")
            return {key: matches[0].get(key) for key in ("id", "root", "max_model_len")}
        except Exception as exc:
            raise ToolInfrastructureError(f"Embedding server verification failed: {exc}") from exc

    def encode(self, texts):
        if not texts:
            raise ValueError("Cannot encode empty batch")
        started = time.monotonic()
        request_id = str(time.time_ns())
        self._event({"event": "start", "request": request_id, "items": len(texts),
                     "input_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()})
        try:
            # The same text can yield different IDs in HF and the server's
            # text preprocessing path. Use the fingerprinted local tokenizer
            # for query strings just as the index builder does for documents.
            if self.input_policy == "local_hf_token_ids_v1" and self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.fingerprint["root"], local_files_only=True)
            if all(isinstance(text, str) for text in texts):
                inputs = [self.tokenizer.encode(text, add_special_tokens=True) for text in texts] if self.tokenizer is not None else texts
            elif all(isinstance(text, list) and text and all(type(token) is int for token in text) for text in texts):
                inputs = texts  # Index windows already include their special tokens.
            else:
                raise ValueError("Embedding input must be homogeneous strings or token-ID lists")
            add_special_tokens = all(isinstance(text, str) for text in inputs)
            request = urllib.request.Request(self.url + "/embeddings", data=json.dumps({
                "model": self.model, "input": inputs, "encoding_format": "float",
                "add_special_tokens": add_special_tokens}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.load(response)
            items = sorted(raw["data"], key=lambda item: item["index"])
            if [item["index"] for item in items] != list(range(len(texts))):
                raise ValueError("Embedding response count/index mismatch")
            matrix = np.asarray([item["embedding"] for item in items], dtype=np.float32)
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError("Nonfinite/malformed embedding")
            norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            if (norm <= 0).any():
                raise ValueError("Embedding has zero norm")
            matrix /= norm
            self._event({"event": "success", "request": request_id, "dimension": matrix.shape[1],
                         "usage": raw.get("usage"), "latency_sec": time.monotonic() - started,
                         "input_policy": self.input_policy,
                         "payload_input_sha256": hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode()).hexdigest(),
                         "add_special_tokens": add_special_tokens})
            return matrix
        except Exception as exc:
            self._event({"event": "error", "request": request_id, "error": str(exc),
                         "latency_sec": time.monotonic() - started})
            raise ToolInfrastructureError(f"Embedding request failed: {exc}") from exc


def _chunks(corpus):
    if corpus._sqlite is not None:
        for row in corpus._sqlite.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id"):
            yield corpus.get_chunk(row[0])
    else:
        yield from (corpus.chunks[cid] for cid in sorted(corpus.chunks))


def build_index(corpus, index_dir, encoder, *, corpus_sha256, batch_size=32,
                window_fn=None, window_policy=None):
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build_index(corpus, root, encoder, corpus_sha256=corpus_sha256,
                            batch_size=batch_size, window_fn=window_fn, window_policy=window_policy)


def _build_index(corpus, index_dir, encoder, *, corpus_sha256, batch_size=32,
                 window_fn=None, window_policy=None):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    for chunk in _chunks(corpus):
        # Public title and paragraph only; never task answer/decomposition.
        text = chunk.title + "\n" + chunk.content
        windows = window_fn(text) if window_fn else [text]
        for window_index, window in enumerate(windows):
            rows.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                         "window": window_index,
                         "input_sha256": hashlib.sha256(json.dumps(window, ensure_ascii=False).encode()).hexdigest()})
            inputs.append(window)
    if not rows:
        raise ValueError("Empty corpus")
    rows_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    spec = {"version": "dense-windows-v2", "corpus_sha256": corpus_sha256,
            "encoder_model": encoder.model, "server": encoder.verify_server(),
            "encoder_fingerprint": getattr(encoder, "fingerprint", None),
            "query_instruction": encoder.query_instruction, "rows_sha256": rows_hash,
            "input_policy": getattr(encoder, "input_policy", "encoder_defined_v1"),
            "window_policy": window_policy or {"mode": "single_input_no_truncation"},
            "rows": len(rows), "paragraphs": len({row["chunk_id"] for row in rows})}
    build_path = root / "build.json"
    if build_path.exists() and json.loads(build_path.read_text()) != spec:
        raise ValueError("Index source/encoder/window configuration changed; use new index directory")
    if (root / "manifest.json").exists():
        existing = json.loads((root / "manifest.json").read_text())
        if existing.get("complete"):
            if any(existing.get(key) != value for key, value in spec.items()):
                raise ValueError("Completed index configuration mismatch")
            if (file_hash(root / "rows.json") != existing["rows_file_sha256"] or
                file_hash(root / "vectors.npy") != existing["vectors_sha256"]):
                raise ValueError("Completed index checksum mismatch; rebuild in a new directory")
            return existing
    save_json(build_path, spec)
    save_json(root / "rows.json", {"rows": rows})
    progress = json.loads((root / "progress.json").read_text()) if (root / "progress.json").exists() else {"rows_done": 0, "batches": []}
    done = progress["rows_done"]
    batches = progress.get("batches", [])
    vectors = np.load(root / "vectors.npy", mmap_mode="r+") if (root / "vectors.npy").exists() else None
    if vectors is not None and vectors.shape[0] != len(rows):
        raise ValueError("Index vector row count mismatch")
    if not 0 <= done <= len(rows) or (done and vectors is None):
        raise ValueError("Invalid index progress")
    verified = 0
    for batch_record in batches:
        start, end = batch_record["start"], batch_record["end"]
        if start != verified or not start < end <= done:
            raise ValueError("Invalid committed batch sequence")
        block_hash = hashlib.sha256(vectors[start:end].tobytes()).hexdigest()
        if block_hash != batch_record["sha256"]:
            raise ValueError("Incomplete index committed-prefix checksum mismatch; use a new index directory")
        verified = end
    if verified != done:
        raise ValueError("Incomplete index lacks committed-prefix checksums")
    for start in range(done, len(inputs), batch_size):
        batch = np.asarray(encoder.encode(inputs[start:start + batch_size]), dtype=np.float32)
        if batch.ndim != 2 or batch.shape[0] != min(batch_size, len(inputs) - start) or not np.isfinite(batch).all():
            raise ValueError("Malformed document embeddings")
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Zero document embedding")
        batch /= norms
        if vectors is None:
            vectors = np.lib.format.open_memmap(root / "vectors.npy", mode="w+", dtype=np.float32,
                                               shape=(len(rows), batch.shape[1]))
        if batch.shape[1] != vectors.shape[1]:
            raise ValueError("Embedding dimension changed")
        end = start + len(batch)
        vectors[start:end] = batch
        vectors.flush()
        with (root / "vectors.npy").open("rb") as handle:
            os.fsync(handle.fileno())
        batches.append({"start": start, "end": end,
                        "sha256": hashlib.sha256(vectors[start:end].tobytes()).hexdigest()})
        save_json(root / "progress.json", {"rows_done": end, "batches": batches})
        print(f"Embedding windows {end}/{len(rows)}", flush=True)
    metadata = {**spec, "complete": True, "dimension": vectors.shape[1], "dtype": "float32",
                "vectors_sha256": file_hash(root / "vectors.npy"),
                "rows_file_sha256": file_hash(root / "rows.json")}
    save_json(root / "manifest.json", metadata)
    return metadata


class DenseCorpus:
    def __init__(self, corpus, index_dir, encoder, mode="hybrid", *, corpus_sha256,
                 candidate_pool=50, rrf_k=60):
        if mode not in {"dense", "hybrid"} or candidate_pool < 1 or rrf_k <= 0:
            raise ValueError("Invalid retrieval configuration")
        self.corpus, self.encoder, self.mode = corpus, encoder, mode
        self.candidate_pool, self.rrf_k = candidate_pool, rrf_k
        root = Path(index_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest.get("complete") or manifest["corpus_sha256"] != corpus_sha256:
            raise ValueError("Incomplete or mismatched corpus index")
        if manifest.get("input_policy") != getattr(encoder, "input_policy", "encoder_defined_v1"):
            raise ValueError("Embedding query tokenization policy mismatch; use a new index directory")
        if (manifest["encoder_model"] != encoder.model or
            manifest["query_instruction"] != encoder.query_instruction or
            manifest["server"] != encoder.verify_server() or
            manifest["encoder_fingerprint"] != getattr(encoder, "fingerprint", None)):
            raise ValueError("Embedding model/configuration mismatch")
        for name, key in [("vectors.npy", "vectors_sha256"), ("rows.json", "rows_file_sha256")]:
            if file_hash(root / name) != manifest[key]:
                raise ValueError("Dense index checksum mismatch")
        self.rows = json.loads((root / "rows.json").read_text())["rows"]
        self.vectors = np.load(root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if self.vectors.shape != (len(self.rows), manifest["dimension"]) or self.vectors.dtype != np.float32:
            raise ValueError("Dense index vector shape/dtype mismatch")
        self.manifest = manifest

    def __getattr__(self, name):
        return getattr(self.corpus, name)

    def __contains__(self, cid):
        return cid in self.corpus

    def __len__(self):
        return len(self.corpus)

    def search_simple(self, query, topk=10, doc_ids=None):
        return self.search(query, topk, doc_ids)

    def search(self, query, topk=10, doc_ids=None):
        if topk <= 0 or doc_ids == []:
            return []
        encoded = self.encoder.encode([self.encoder.query_instruction + query])[0]
        if encoded.shape != (self.vectors.shape[1],):
            raise ToolInfrastructureError("Query/document embedding dimension mismatch")
        scores = self.vectors @ encoded
        allowed = set(doc_ids) if doc_ids is not None else None
        paragraph_scores = {}
        for row, score in zip(self.rows, scores):
            if allowed is not None and row["doc_id"] not in allowed:
                continue
            cid = row["chunk_id"]
            paragraph_scores[cid] = max(paragraph_scores.get(cid, -float("inf")), float(score))
        pool = max(topk, self.candidate_pool)
        dense = sorted(paragraph_scores, key=lambda cid: (-paragraph_scores[cid], cid))[:pool]
        if self.mode == "hybrid":
            lexical = self.corpus.search(query, pool, doc_ids)
            fused = defaultdict(float)
            for ranked in (dense, [c.chunk_id for c in lexical]):
                for rank, cid in enumerate(ranked, 1):
                    fused[cid] += 1 / (self.rrf_k + rank)
            selected = sorted(fused, key=lambda cid: (-fused[cid], cid))[:topk]
            result_scores = fused
        else:
            selected, result_scores = dense[:topk], paragraph_scores
        results = []
        for rank, cid in enumerate(selected, 1):
            chunk = self.corpus.get_chunk(cid)
            results.append(CandidateChunk(cid, chunk.doc_id, result_scores[cid], rank, query,
                                          chunk.title, chunk.content[:200]))
        return results
