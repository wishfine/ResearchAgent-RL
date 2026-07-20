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


def _install_disk_reload_compat() -> None:
    """Bridge Vime's new disk-reload RPC call to vLLM 0.23's old signature.

    Vime's checkpoint directory is already monotonically versioned
    (``weight_v000001``, ...), so omitting the unsupported ``weight_version``
    keyword cannot cause a stale checkpoint to be selected.  vLLM 0.23 also
    acknowledges the controller request before every TP worker has closed the
    safetensors shards.  Therefore the current checkpoint is never removed
    here; only a version two successful rollout generations behind is pruned.
    This deliberately patches only the trainer-side dispatcher and leaves the
    actual vLLM disk loader untouched.
    """

    module = importlib.import_module(
        "vime.backends.megatron_utils.update_weight.update_weight_from_disk"
    )
    updater_cls = module.UpdateWeightFromDisk
    original = updater_cls.update_weights
    if getattr(original, "_research_agent_disk_compat", False):
        return

    @module.torch.no_grad()
    def update_weights_without_weight_version(self: Any) -> None:
        self.weight_version += 1
        version_dir = module.Path(self.args.update_weight_disk_dir) / (
            f"weight_v{self.weight_version:06d}"
        )
        if module.dist.get_rank() == 0:
            module.shutil.rmtree(version_dir, ignore_errors=True)
        module.dist.barrier(group=module.get_gloo_group())

        if module.dist.get_rank() == 0:
            module.logger.info(
                "Updating rollout weights from disk checkpoint %s (vLLM 0.23 compatibility mode)",
                version_dir,
            )
            module.ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            module.ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        module.dist.barrier(group=module.get_gloo_group())

        module.save_hf_model_to_path(
            self.args,
            version_dir,
            self.model,
            model_name=self.model_name,
            quantization_config=self.quantization_config,
            progress_desc="Save HF weights for update from disk",
        )
        module.dist.barrier(group=module.get_gloo_group())

        refs = []
        if module.dist.get_rank() == 0:
            # vLLM 0.23 accepts ``model_path`` but not Vime's newer
            # ``weight_version`` keyword.  Each invocation uses a unique path.
            refs = [
                engine.update_weights_from_disk.remote(model_path=str(version_dir))
                for engine in self.rollout_engines
            ]
        module.ray.get(refs)

        # Do not use Vime's immediate cleanup here.  On this vLLM release the
        # HTTP response can arrive before every TP worker has opened all shard
        # files, which makes deleting ``version_dir`` a data race.  A version
        # older than ``keep_last`` has been superseded by a later checkpoint
        # that has already served a complete rollout before this next update.
        keep_last = int(os.environ.get("VIME_DISK_WEIGHT_SYNC_KEEP_LAST", "2"))
        stale_version = self.weight_version - keep_last
        if stale_version >= 1 and module.dist.get_rank() == 0:
            stale_dir = module.Path(self.args.update_weight_disk_dir) / (
                f"weight_v{stale_version:06d}"
            )
            if stale_dir.exists():
                module.shutil.rmtree(stale_dir, ignore_errors=True)
                module.logger.info(
                    "Pruned superseded disk checkpoint %s; retaining last %d versions",
                    stale_dir,
                    keep_last,
                )
        module.ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        module.dist.barrier(group=module.get_gloo_group())

    update_weights_without_weight_version._research_agent_disk_compat = True
    updater_cls.update_weights = update_weights_without_weight_version
    print(
        "[RA_DISK_WEIGHT_SYNC_COMPAT] installed: omit unsupported weight_version keyword",
        flush=True,
    )


if os.environ.get("VIME_WEIGHT_SYNC_TRACE") == "1":
    _install()

if os.environ.get("VIME_DISK_WEIGHT_SYNC_COMPAT") == "1":
    _install_disk_reload_compat()
