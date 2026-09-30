#!/usr/bin/env python3
"""Check BGE CLS / Qwen3 LAST service against local HF pooling; no downloads."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--embedding_url", default="http://127.0.0.1:8106/v1")
    parser.add_argument("--embedding_model", default="bge-small-zh-v1.5")
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    from research_agent.core.corpus.dense import EmbeddingClient
    texts = ["The river flows past the city of Perm.", "The river flows past the city of Perm.",
             "An unrelated football stadium."]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True,
                                     torch_dtype=torch.float32, attn_implementation="eager").eval().cpu()
    pooling = "CLS" if model.config.model_type == "bert" else "LAST" if model.config.model_type == "qwen3" else None
    assert pooling is not None, "Only BGE/BERT and Qwen3 supported"
    if pooling == "LAST":
        tokenizer.padding_side = "left"
    with torch.inference_mode():
        result = model(**tokenizer(texts, padding=True, truncation=False, return_tensors="pt"))
        hidden = result.last_hidden_state[:, 0 if pooling == "CLS" else -1].float()
        reference = torch.nn.functional.normalize(hidden, dim=1).numpy()
    client = EmbeddingClient(args.embedding_url, args.embedding_model)
    print("server:", client.verify_server())
    actual = client.encode(texts)
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    agreement = (actual * reference).sum(axis=1)
    assert np.all(agreement > 0.999), f"Backend/tokenization/CLS mismatch: {agreement}"
    pretokenized = client.encode([tokenizer.encode(text, add_special_tokens=True) for text in texts])
    assert np.all((pretokenized * actual).sum(axis=1) > 0.999), "Pretokenized-window mismatch"
    print("dimension:", actual.shape[1])
    print("HF", pooling, "vs vLLM cosine:", agreement.tolist())
    print("Embedding backend: OK (not retrieval-quality evidence)")


if __name__ == "__main__":
    main()
