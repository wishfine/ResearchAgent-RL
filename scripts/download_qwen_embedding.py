#!/usr/bin/env python3
"""Explicit optional Qwen embedding download. Never installs dependencies."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    repo = "Qwen/Qwen3-Embedding-0.6B"
    receipt = args.model_dir / "download_receipt.json"
    if args.model_dir.exists() and any(args.model_dir.iterdir()) and not receipt.exists():
        parser.error("Nonempty model directory has no download receipt. Reuse it or choose a new directory.")
    revision = HfApi().model_info(repo, revision=args.revision).sha if not receipt.exists() else json.loads(receipt.read_text())["revision"]
    args.model_dir.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"repo": repo, "revision": revision,
                                  "directory": str(args.model_dir.resolve()), "complete": False}, indent=2) + "\n")
    path = snapshot_download(repo, revision=revision, local_dir=args.model_dir,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*/config.json"])
    receipt.write_text(json.dumps({"repo": repo, "revision": revision,
                                  "directory": str(args.model_dir.resolve()), "complete": True}, indent=2) + "\n")
    print(path, "revision=", revision)


if __name__ == "__main__":
    main()
