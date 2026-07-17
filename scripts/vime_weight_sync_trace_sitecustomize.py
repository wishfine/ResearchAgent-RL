"""Optional startup hook for tracing Vime's outgoing weight-update metadata.

The smoke launcher copies this file to ``sitecustomize.py`` only when
``VIME_WEIGHT_SYNC_TRACE=1``.  Python imports that special module before the
Ray workers load Vime, allowing us to observe the trainer-to-vLLM boundary
without editing the separately managed Vime checkout.
"""

from __future__ import annotations

import importlib
import os
from collections import Counter
from typing import Any


def _summarize(names: list[str]) -> str:
    prefixes = Counter(name.split(".", 1)[0] for name in names)
    prefix_summary = ",".join(f"{prefix}:{count}" for prefix, count in sorted(prefixes.items()))
    head = ";".join(names[:3])
    tail = ";".join(names[-3:])
    return f"count={len(names)} prefixes={prefix_summary} first=[{head}] last=[{tail}]"


def _install() -> None:
    module = importlib.import_module(
        "vime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    )
    original = module.update_weights_from_distributed
    if getattr(original, "_research_agent_trace", False):
        return

    def traced_update_weights_from_distributed(*args: Any, **kwargs: Any) -> Any:
        tensors = kwargs.get("converted_named_tensors")
        if tensors is None and len(args) >= 5:
            tensors = args[4]
        raw_tensors = list(tensors or [])
        raw_names = [name for name, _ in raw_tensors]
        mode = os.environ.get("VIME_WEIGHT_SYNC_NAME_MODE", "hf")
        if mode == "vllm_native":
            mapped_tensors = []
            for name, tensor in raw_tensors:
                if name.startswith("model.language_model."):
                    name = "language_model.model." + name.removeprefix("model.language_model.")
                elif name.startswith("lm_head."):
                    name = "language_model." + name
                mapped_tensors.append((name, tensor))
        else:
            mapped_tensors = raw_tensors

        sent_names = [name for name, _ in mapped_tensors]
        print(
            "[RA_WEIGHT_SYNC_TRAINER] "
            f"mode={mode} raw=({_summarize(raw_names)}) sent=({_summarize(sent_names)})",
            flush=True,
        )
        if "converted_named_tensors" in kwargs:
            kwargs["converted_named_tensors"] = mapped_tensors
            return original(*args, **kwargs)
        return original(*args[:4], mapped_tensors, *args[5:], **kwargs)

    traced_update_weights_from_distributed._research_agent_trace = True
    module.update_weights_from_distributed = traced_update_weights_from_distributed


if os.environ.get("VIME_WEIGHT_SYNC_TRACE") == "1":
    _install()
