#!/usr/bin/env python3
"""Materialize a standalone HF checkpoint from Vime's text-only weight export.

Vime trains and exports the Qwen3.5 language model, while the original HF
checkpoint also contains a frozen visual tower.  vLLM can hot-reload the
language-only export into an already-loaded base model, but it cannot start a
fresh server from that partial directory.  This tool writes a complete HF
checkpoint by using every base tensor except those explicitly supplied by the
trained export.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _find_index(directory: Path) -> Path:
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError(
            f"Expected exactly one '*.safetensors.index.json' in {directory}, found {indexes}"
        )
    return indexes[0]


def _load_index(directory: Path) -> tuple[Path, dict[str, Any]]:
    index_path = _find_index(directory)
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index without a non-empty weight_map: {index_path}")
    missing_files = sorted(
        filename for filename in set(weight_map.values()) if not (directory / filename).is_file()
    )
    if missing_files:
        raise FileNotFoundError(f"Missing shard(s) declared by {index_path}: {missing_files}")
    return index_path, index


def _copy_auxiliary_files(base_dir: Path, output_dir: Path, *, index_name: str) -> None:
    for source in base_dir.iterdir():
        if source.name == index_name or source.suffix == ".safetensors":
            continue
        target = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def materialize_checkpoint(
    *,
    base_dir: Path,
    trained_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Overlay trained tensors onto a full base checkpoint and return counts."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = base_dir.resolve()
    trained_dir = trained_dir.resolve()
    output_dir = output_dir.resolve()
    if not base_dir.is_dir() or not trained_dir.is_dir():
        raise FileNotFoundError("base_dir and trained_dir must both be directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_index_path, base_index = _load_index(base_dir)
    _, trained_index = _load_index(trained_dir)
    base_map: Mapping[str, str] = base_index["weight_map"]
    trained_map: Mapping[str, str] = trained_index["weight_map"]

    unexpected = sorted(set(trained_map) - set(base_map))
    if unexpected:
        preview = ", ".join(unexpected[:10])
        raise ValueError(f"Trained export has tensors absent from base checkpoint: {preview}")

    _copy_auxiliary_files(base_dir, output_dir, index_name=base_index_path.name)

    base_handles: dict[str, Any] = {}
    trained_handles: dict[str, Any] = {}

    def read_tensor(directory: Path, filename: str, name: str, cache: dict[str, Any]):
        path = directory / filename
        handle = cache.get(filename)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            cache[filename] = handle
        return handle.get_tensor(name)

    try:
        for filename in sorted(set(base_map.values())):
            names = sorted(name for name, shard in base_map.items() if shard == filename)
            tensors = {}
            for name in names:
                if name in trained_map:
                    tensors[name] = read_tensor(
                        trained_dir, trained_map[name], name, trained_handles
                    )
                else:
                    tensors[name] = read_tensor(base_dir, filename, name, base_handles)
            # Same shard names and tensor shapes preserve the base index layout.
            save_file(tensors, output_dir / filename, metadata={"format": "pt"})
    finally:
        for handle in [*base_handles.values(), *trained_handles.values()]:
            close = getattr(handle, "close", None)
            if close is not None:
                close()

    with (output_dir / base_index_path.name).open("w", encoding="utf-8") as handle:
        json.dump(base_index, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return {
        "base_tensors": len(base_map),
        "trained_tensors_overlaid": len(trained_map),
        "base_only_tensors_retained": len(base_map) - len(trained_map),
        "shards_written": len(set(base_map.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_dir", type=Path, required=True)
    parser.add_argument("--trained_weights_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = materialize_checkpoint(
        base_dir=args.base_model_dir,
        trained_dir=args.trained_weights_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("[SUCCESS] Materialized standalone HF checkpoint")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
