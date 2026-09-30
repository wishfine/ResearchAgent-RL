from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

safetensors = pytest.importorskip("safetensors")
torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "materialize_vime_hf_checkpoint.py"
    spec = importlib.util.spec_from_file_location("materialize_vime_hf_checkpoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_index(directory: Path, mapping: dict[str, str]) -> None:
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": mapping}),
        encoding="utf-8",
    )


def test_materialize_overlays_language_and_keeps_visual_tensors(tmp_path: Path) -> None:
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    output = tmp_path / "output"
    base.mkdir()
    trained.mkdir()

    save_file(
        {
            "language_model.weight": torch.tensor([1.0]),
            "visual.tower.weight": torch.tensor([2.0]),
        },
        base / "model-00001-of-00001.safetensors",
        metadata={"format": "pt"},
    )
    _write_index(
        base,
        {
            "language_model.weight": "model-00001-of-00001.safetensors",
            "visual.tower.weight": "model-00001-of-00001.safetensors",
        },
    )
    (base / "config.json").write_text('{"model_type": "toy"}', encoding="utf-8")

    save_file(
        {"language_model.weight": torch.tensor([9.0])},
        trained / "model-00001-of-00001.safetensors",
        metadata={"format": "pt"},
    )
    _write_index(trained, {"language_model.weight": "model-00001-of-00001.safetensors"})

    module = _load_module()
    stats = module.materialize_checkpoint(
        base_dir=base, trained_dir=trained, output_dir=output
    )

    weights = load_file(output / "model-00001-of-00001.safetensors")
    assert weights["language_model.weight"].item() == 9.0
    assert weights["visual.tower.weight"].item() == 2.0
    assert (output / "config.json").is_file()
    assert stats == {
        "base_tensors": 2,
        "trained_tensors_overlaid": 1,
        "base_only_tensors_retained": 1,
        "shards_written": 1,
    }
