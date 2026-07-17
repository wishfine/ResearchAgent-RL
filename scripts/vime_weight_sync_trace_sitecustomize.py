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
        names = [name for name, _ in tensors] if tensors is not None else []
        print(f"[RA_WEIGHT_SYNC_TRAINER] {_summarize(names)}", flush=True)
        return original(*args, **kwargs)

    traced_update_weights_from_distributed._research_agent_trace = True
    module.update_weights_from_distributed = traced_update_weights_from_distributed


if os.environ.get("VIME_WEIGHT_SYNC_TRACE") == "1":
    _install()
