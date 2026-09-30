#!/usr/bin/env python3
"""Build a split-isolated vector index using an existing embedding service.

Use --model_dir for local-only tokenizer/model fingerprinting. This script
does not download a model or start/stop a GPU service.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.corpus.dense import EmbeddingClient, build_index, file_hash


def model_fingerprint(model_dir, pooling="auto"):
    root = Path(model_dir).resolve()
    if not (root / "config.json").is_file():
        raise ValueError("Missing local embedding config.json")
    config = json.loads((root / "config.json").read_text())
    model_type = config.get("model_type")
    expected = "CLS" if model_type == "bert" else "LAST" if model_type == "qwen3" else None
    if expected is None or pooling not in {"auto", expected}:
        raise ValueError("Only BGE/BERT CLS and Qwen3 LAST are supported by this builder")
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}}
    if not any(name.endswith((".bin", ".safetensors")) for name in files):
        raise ValueError("Embedding model weights missing")
    return {"root": str(root), "files": files, "pooling": expected, "normalize": True}


def token_windows(tokenizer, text, max_length=512, overlap=64):
    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= overlap < content_length:
        raise ValueError("Invalid embedding window overlap")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    starts = range(0, max(1, len(tokens)), content_length - overlap)
    result = []
    for start in starts:
        window = tokenizer.build_inputs_with_special_tokens(tokens[start:start + content_length])
        result.append(window)
        if start + content_length >= len(tokens):
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--embedding_url", default="http://127.0.0.1:8106/v1")
    parser.add_argument("--embedding_model", default="bge-small-zh-v1.5")
    parser.add_argument("--query_instruction", default=None)
    parser.add_argument("--pooling", choices=["auto", "CLS", "LAST"], default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    fingerprint = model_fingerprint(args.model_dir, args.pooling)
    config = json.loads((args.model_dir / "config.json").read_text())
    args.max_length = args.max_length or (512 if fingerprint["pooling"] == "CLS" else 2048)
    if args.max_length > config["max_position_embeddings"]:
        parser.error("Embedding window exceeds model max_position_embeddings")
    if args.query_instruction is None:
        args.query_instruction = "" if fingerprint["pooling"] == "CLS" else (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    corpus = CorpusStore(str(args.corpus_dir))
    corpus.load()
    if corpus._sqlite is None:
        parser.error("Provide the fixed MuSiQue SQLite corpus")
    encoder = EmbeddingClient(args.embedding_url, args.embedding_model, args.query_instruction,
                              event_log=args.index_dir / "embedding_cost.jsonl", fingerprint=fingerprint)
    try:
        manifest = build_index(corpus, args.index_dir, encoder,
                               corpus_sha256=file_hash(args.corpus_dir / "corpus.sqlite"),
                               batch_size=args.batch_size,
                               window_fn=lambda text: token_windows(tokenizer, text, args.max_length, args.overlap),
                               window_policy={"mode": "token_windows", "max_length": args.max_length,
                                              "overlap": args.overlap, "paragraph_score": "max_window_cosine"})
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        corpus.close()


if __name__ == "__main__":
    main()
