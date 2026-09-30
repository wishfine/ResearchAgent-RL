#!/usr/bin/env python3
"""Paired continuations for distinct SEARCH actions with identical retrieval IDs.

Run in three phases: prepare a label-blind pair manifest from first-search
samples; collect independent frozen-policy continuations; analyze reward gaps.
This diagnoses abstraction error. It does not optimize a policy or establish
a trust-region guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.baselines.llm_actor import ActorTurn, LLMActor
from research_agent.core.baseline_runner import run_episode
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.effect_space import effect_key
from research_agent.core.env.llm_client import LLMClient
from research_agent.core.schema.parser import ActionParser
from scripts.run_llm_baseline import (
    chat_endpoint,
    load_task,
    make_env,
    models_endpoint,
    wait_for_model_server,
)

PROTOCOL = "effect-space-paired-continuation-v1"
MODES = ("original", "canonical")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1], check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def selected_task_sha256(tasks_dir: Path, task_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for task_id in sorted(task_ids):
        path = tasks_dir / f"{task_id}.json"
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def served_model_ids(model_url: str) -> list[str]:
    with urllib.request.urlopen(models_endpoint(model_url), timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    ids = [item["id"] for item in payload.get("data", [])
           if isinstance(item, dict) and isinstance(item.get("id"), str)]
    if not ids:
        raise RuntimeError("Model server returned no model IDs")
    return ids


def surface_key(query: str, topk: int) -> tuple[str, int]:
    return " ".join(query.casefold().split()), topk


def validated_sample(row: dict) -> tuple[str, int] | None:
    if not row.get("valid_search"):
        return None
    query, topk, raw = row.get("query"), row.get("topk"), row.get("raw_content")
    if (not isinstance(query, str) or not query.strip()
            or not isinstance(topk, int) or isinstance(topk, bool)
            or not 1 <= topk <= 100 or not isinstance(raw, str)):
        raise ValueError(f"Malformed valid SEARCH sample: {row.get('task_id')!r}")
    parsed = ActionParser.parse(raw)
    if parsed.tool != "SEARCH" or parsed.params.get("query") != query:
        raise ValueError(f"Saved raw action disagrees with query: {row.get('task_id')!r}")
    if parsed.params.get("topk", 10) != topk:
        raise ValueError(f"Saved raw action disagrees with topk: {row.get('task_id')!r}")
    valid, error = parsed.validate()
    if not valid:
        raise ValueError(f"Invalid saved SEARCH action: {error}")
    return query, topk


def prepare_manifest(samples: list[dict], tasks_dir: Path, corpus: CorpusStore, *,
                     max_tasks: int, max_pairs_per_task: int, seed: int) -> dict:
    """Select pairs without reading ground-truth answers or citations."""
    if max_tasks <= 0 or max_pairs_per_task <= 0:
        raise ValueError("max_tasks and max_pairs_per_task must be positive")
    by_task: dict[str, list[dict]] = defaultdict(list)
    seen_samples: set[tuple[str, int]] = set()
    for row in samples:
        task_id, sample_index = row.get("task_id"), row.get("sample_index")
        if (not isinstance(task_id, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id)
                or not isinstance(sample_index, int) or isinstance(sample_index, bool)):
            raise ValueError("Invalid task_id or sample_index in samples")
        key = (task_id, sample_index)
        if key in seen_samples:
            raise ValueError(f"Duplicate first-search sample: {key}")
        seen_samples.add(key)
        by_task[task_id].append(row)

    eligible = []
    for task_id in sorted(by_task):
        task_path = tasks_dir / f"{task_id}.json"
        if not task_path.is_file():
            raise ValueError(f"Missing task: {task_path}")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("retrieval_scope") != "split_corpus" or task.get("reference_docs"):
            raise ValueError(f"{task_id}: pilot requires split-wide unscoped retrieval")
        distinct_surfaces: dict[tuple[str, int], dict] = {}
        for row in sorted(by_task[task_id], key=lambda item: item["sample_index"]):
            parsed = validated_sample(row)
            if parsed is None:
                continue
            query, topk = parsed
            key = surface_key(query, topk)
            if key in distinct_surfaces:
                continue
            candidates = corpus.search(query, topk=topk, doc_ids=None)
            effect = effect_key(item.chunk_id for item in candidates)
            if not effect:
                continue  # Empty retrieval is not useful for the first pilot.
            distinct_surfaces[key] = {
                "sample_index": row["sample_index"], "query": query, "topk": topk,
                "raw_content": row["raw_content"], "effect_chunk_ids": list(effect),
                "candidate_scores": [float(item.score) for item in candidates],
            }
        variants = list(distinct_surfaces.values())
        by_effect: dict[tuple[tuple[str, ...], int], list[dict]] = defaultdict(list)
        for variant in variants:
            by_effect[(tuple(variant["effect_chunk_ids"]), variant["topk"])].append(variant)
        within = [
            (a["sample_index"], b["sample_index"])
            for group in by_effect.values() for a, b in itertools.combinations(group, 2)
        ]
        if not within:
            continue
        between = [
            (a["sample_index"], b["sample_index"])
            for a, b in itertools.combinations(variants, 2)
            if a["topk"] == b["topk"]
            and a["effect_chunk_ids"] != b["effect_chunk_ids"]
        ]
        eligible.append((task_id, variants, within, between))

    if not eligible:
        raise ValueError("No distinct-query same-effect pairs; run or expand the first-search pilot")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = []
    for task_id, variants, within, between in eligible[:max_tasks]:
        rng.shuffle(within)
        rng.shuffle(between)
        selected_within = within[:max_pairs_per_task]
        selected_between = between[:min(len(between), len(selected_within))]
        pairs = [
            {"kind": kind, "left": left, "right": right}
            for kind, selected in (("within", selected_within), ("between", selected_between))
            for left, right in selected
        ]
        needed = {index for pair in pairs for index in (pair["left"], pair["right"])}
        chosen.append({
            "task_id": task_id,
            "variants": sorted(
                (variant for variant in variants if variant["sample_index"] in needed),
                key=lambda variant: variant["sample_index"],
            ),
            "pairs": pairs,
        })
    return {
        "protocol": PROTOCOL,
        "selection_seed": seed,
        "eligible_tasks": len(eligible),
        "selected_tasks": len(chosen),
        "tasks": chosen,
        "selection_uses_labels": False,
        "limitations": [
            "Only first SEARCH from a shared prefix is tested.",
            "Same ordered retrieved IDs are not guaranteed to be the same post-tool state.",
            "Pairs are selected without labels; labels are used only by offline episode scoring.",
        ],
    }


def canonical_action_text(query: str, topk: int) -> str:
    payload = {
        "tool": "SEARCH",
        "intent": "Find evidence for the question",
        "params": {"query": query, "topk": topk},
    }
    return f"<action>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</action>"


class ForcedFirstSearchActor:
    """Execute a saved first action while controlling what enters later history."""

    def __init__(self, base_actor: LLMActor, variant: dict, history_text: str):
        self.base_actor = base_actor
        self.variant = variant
        self.history_text = history_text
        self.first = True

    def decide(self, prompt: str, **kwargs) -> ActorTurn:
        if not self.first:
            return self.base_actor.decide(prompt, **kwargs)
        self.first = False
        action = ActionParser.parse(self.variant["raw_content"])
        if (action.tool != "SEARCH"
                or action.params.get("query") != self.variant["query"]
                or action.params.get("topk", 10) != self.variant["topk"]):
            raise ValueError("Forced SEARCH disagrees with prepared manifest")
        return ActorTurn(
            raw_content=self.history_text, action=action,
            reasoning=action.reasoning or "", latency_sec=0.0,
            prompt_tokens=0, completion_tokens=0,
        )


def run_variant(task_path: Path, corpus: CorpusStore, base_actor: LLMActor,
                variant: dict, *, history_mode: str, canonical_query: str,
                max_steps: int, max_tokens: int, temperature: float,
                top_p: float) -> dict:
    if history_mode not in MODES:
        raise ValueError(f"Unknown history mode: {history_mode}")
    history = (variant["raw_content"] if history_mode == "original"
               else canonical_action_text(canonical_query, variant["topk"]))
    actor = ForcedFirstSearchActor(base_actor, variant, history)
    result = run_episode(
        make_env(corpus, max_steps), load_task(str(task_path)), actor,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
    )
    if result["termination_reason"] == "actor_error":
        raise RuntimeError(f"Model continuation failed: {result['steps'][-1]['error']}")
    first = result["steps"][0]
    if first["action"]["tool"] != "SEARCH" or not first["tool_result"]["success"]:
        raise RuntimeError("Forced first SEARCH did not execute successfully")
    observed = [item["chunk_id"] for item in first["tool_result"]["data"]["candidates"]]
    if observed != variant["effect_chunk_ids"]:
        raise RuntimeError(
            "Retrieval effect changed since pair preparation; rerun prepare with the current corpus"
        )
    return result


def collect_task_continuations(task_spec: dict, task_path: Path, corpus: CorpusStore,
                               actor: LLMActor, *, repeats: int, seed: int,
                               max_steps: int, max_tokens: int,
                               temperature: float, top_p: float) -> dict:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    variants = task_spec["variants"]
    canonical = {}
    for variant in variants:
        key = (tuple(variant["effect_chunk_ids"]), variant["topk"])
        canonical.setdefault(key, variant["query"])
    rng = random.Random(seed)
    runs = []
    for repeat in range(repeats):
        for history_mode in MODES:
            shuffled = list(variants)
            rng.shuffle(shuffled)
            for variant in shuffled:
                key = (tuple(variant["effect_chunk_ids"]), variant["topk"])
                record = run_variant(
                    task_path, corpus, actor, variant,
                    history_mode=history_mode, canonical_query=canonical[key],
                    max_steps=max_steps, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p,
                )
                runs.append({
                    "sample_index": variant["sample_index"],
                    "repeat": repeat,
                    "history_mode": history_mode,
                    "record": record,
                })
    return {"task_id": task_spec["task_id"], "runs": runs}


def metric_value(record: dict, field: str) -> float:
    return float(record[field] if field == "reward" else record["eval_metrics"][field])


def bootstrap_mean_ci(values: list[float], *, seed: int, draws: int = 2000) -> list[float] | None:
    """Task-level percentile interval; descriptive, not a causal confidence bound."""
    if len(values) < 5:
        return None
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[int(0.975 * draws)]]


def analyze_continuations(manifest: dict, task_outputs: dict[str, dict]) -> dict:
    pair_records = []
    per_task = []
    for spec in manifest["tasks"]:
        task_id = spec["task_id"]
        result = task_outputs.get(task_id)
        if result is None:
            continue
        by_variant: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for run in result["runs"]:
            by_variant[(run["sample_index"], run["history_mode"])].append(
                run["record"]
            )
        noise_floor = {}
        for mode in MODES:
            split_gaps = []
            for (sample_index, sample_mode), records in by_variant.items():
                if sample_mode != mode or len(records) < 4:
                    continue
                # This is a split-half Monte Carlo noise diagnostic, not a
                # confidence bound on the unobserved continuation Q value.
                left = [metric_value(item, "reward") for item in records[::2]]
                right = [metric_value(item, "reward") for item in records[1::2]]
                split_gaps.append(abs(sum(left) / len(left) - sum(right) / len(right)))
            noise_floor[mode] = (
                sum(split_gaps) / len(split_gaps) if split_gaps else None
            )
        task_pairs = []
        for pair in spec["pairs"]:
            metrics = {}
            for mode in MODES:
                left = by_variant.get((pair["left"], mode), [])
                right = by_variant.get((pair["right"], mode), [])
                if not left or not right or len(left) != len(right):
                    raise ValueError(f"{task_id}: incomplete or unbalanced runs for {pair}")
                for name, field in (
                    ("reward", "reward"),
                    ("answer_quality", "answer_quality"),
                    ("citation_recall", "citation_recall"),
                    ("task_success", "task_success"),
                ):
                    left_mean = sum(metric_value(item, field) for item in left) / len(left)
                    right_mean = sum(metric_value(item, field) for item in right) / len(right)
                    metrics[f"{mode}_{name}_gap"] = abs(left_mean - right_mean)
            item = {
                "task_id": task_id, "kind": pair["kind"],
                "left": pair["left"], "right": pair["right"],
                "repeats": len(by_variant[(pair["left"], "original")]),
                **metrics,
            }
            if pair["kind"] == "within":
                variants = {variant["sample_index"]: variant
                            for variant in spec.get("variants", [])}
                a, b = variants.get(pair["left"]), variants.get(pair["right"])
                if a is not None and b is not None:
                    item["max_retrieval_score_shift"] = max(
                        (abs(x - y) for x, y in zip(
                            a["candidate_scores"], b["candidate_scores"], strict=True
                        )),
                        default=0.0,
                    )
            pair_records.append(item)
            task_pairs.append(item)
        per_task.append({
            "task_id": task_id,
            "original_split_half_reward_noise_floor": noise_floor["original"],
            "canonical_split_half_reward_noise_floor": noise_floor["canonical"],
            "within_pairs": sum(item["kind"] == "within" for item in task_pairs),
            "between_pairs": sum(item["kind"] == "between" for item in task_pairs),
            "within_original_reward_gap": (
                sum(item["original_reward_gap"] for item in task_pairs
                    if item["kind"] == "within")
                / sum(item["kind"] == "within" for item in task_pairs)
                if any(item["kind"] == "within" for item in task_pairs) else None
            ),
            "within_canonical_reward_gap": (
                sum(item["canonical_reward_gap"] for item in task_pairs
                    if item["kind"] == "within")
                / sum(item["kind"] == "within" for item in task_pairs)
                if any(item["kind"] == "within" for item in task_pairs) else None
            ),
            "between_original_reward_gap": (
                sum(item["original_reward_gap"] for item in task_pairs
                    if item["kind"] == "between")
                / sum(item["kind"] == "between" for item in task_pairs)
                if any(item["kind"] == "between" for item in task_pairs) else None
            ),
        })

    def mean_gap(kind: str, field: str) -> float | None:
        values = [item[field] for item in pair_records if item["kind"] == kind]
        return sum(values) / len(values) if values else None

    paired_between_minus_within = [
        item["between_original_reward_gap"] - item["within_original_reward_gap"]
        for item in per_task if item["between_original_reward_gap"] is not None
        and item["within_original_reward_gap"] is not None
    ]
    within_original_minus_canonical = [
        item["within_original_reward_gap"] - item["within_canonical_reward_gap"]
        for item in per_task if item["within_original_reward_gap"] is not None
        and item["within_canonical_reward_gap"] is not None
    ]
    return {
        "protocol": PROTOCOL,
        "manifest_tasks": len(manifest["tasks"]),
        "completed_tasks": len(per_task),
        "complete": len(per_task) == len(manifest["tasks"]),
        "within_pairs": sum(item["kind"] == "within" for item in pair_records),
        "between_pairs": sum(item["kind"] == "between" for item in pair_records),
        "mean_within_original_reward_gap": mean_gap("within", "original_reward_gap"),
        "mean_within_canonical_reward_gap": mean_gap("within", "canonical_reward_gap"),
        "mean_between_original_reward_gap": mean_gap("between", "original_reward_gap"),
        "mean_within_original_success_gap": mean_gap("within", "original_task_success_gap"),
        "mean_within_canonical_success_gap": mean_gap("within", "canonical_task_success_gap"),
        "mean_original_split_half_reward_noise_floor": (
            sum(item["original_split_half_reward_noise_floor"] for item in per_task
                if item["original_split_half_reward_noise_floor"] is not None)
            / sum(item["original_split_half_reward_noise_floor"] is not None
                  for item in per_task)
            if any(item["original_split_half_reward_noise_floor"] is not None
                   for item in per_task) else None
        ),
        "mean_canonical_split_half_reward_noise_floor": (
            sum(item["canonical_split_half_reward_noise_floor"] for item in per_task
                if item["canonical_split_half_reward_noise_floor"] is not None)
            / sum(item["canonical_split_half_reward_noise_floor"] is not None
                  for item in per_task)
            if any(item["canonical_split_half_reward_noise_floor"] is not None
                   for item in per_task) else None
        ),
        "paired_tasks_with_between": len(paired_between_minus_within),
        "mean_paired_between_minus_within": (
            sum(paired_between_minus_within) / len(paired_between_minus_within)
            if paired_between_minus_within else None
        ),
        "paired_between_minus_within_bootstrap_ci95": bootstrap_mean_ci(
            paired_between_minus_within, seed=manifest.get("selection_seed", 42)
        ),
        "mean_within_original_minus_canonical": (
            sum(within_original_minus_canonical) / len(within_original_minus_canonical)
            if within_original_minus_canonical else None
        ),
        "within_original_minus_canonical_bootstrap_ci95": bootstrap_mean_ci(
            within_original_minus_canonical, seed=manifest.get("selection_seed", 42) + 1
        ),
        "per_task": per_task,
        "pairs": pair_records,
        "limitations": [
            "Reward gaps are descriptive finite-sample estimates, not upper bounds on true Q residuals.",
            "Canonical history is an intervention, not the deployed Agent environment.",
            "Independent model samples have no common random numbers; sampling noise remains.",
            "Split-half reward gaps are a noisy reference scale, not a de-biased Q residual estimate.",
            "Between-effect pairs are a descriptive comparison, not a matched causal control.",
            "Only first SEARCH is studied; later-hop state distributions may differ.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tasks_dir", type=Path)
    parser.add_argument("--corpus_dir", type=Path)
    parser.add_argument("--samples_file", type=Path)
    parser.add_argument("--max_tasks", type=int, default=10)
    parser.add_argument("--max_pairs_per_task", type=int, default=1)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--model_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", default="Qwen3.5-9B")
    parser.add_argument("--model_revision",
                        help="Immutable checkpoint path or revision used by the model service")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = args.output_dir
    manifest_path = output / "manifest.json"
    if args.mode == "prepare":
        if not args.tasks_dir or not args.corpus_dir or not args.samples_file:
            parser.error("prepare requires --tasks_dir, --corpus_dir, --samples_file")
        if manifest_path.exists():
            parser.error(f"Manifest exists: {manifest_path}; choose a fresh output_dir")
        if not args.samples_file.is_file():
            parser.error(f"Missing first-search samples: {args.samples_file}")
        corpus = CorpusStore(str(args.corpus_dir))
        corpus.load()
        if not len(corpus):
            parser.error(f"Empty corpus: {args.corpus_dir}")
        try:
            manifest = prepare_manifest(
                read_jsonl(args.samples_file), args.tasks_dir, corpus,
                max_tasks=args.max_tasks, max_pairs_per_task=args.max_pairs_per_task,
                seed=args.selection_seed,
            )
        finally:
            corpus.close()
        manifest["source"] = {
            "samples_file": str(args.samples_file.resolve()),
            "samples_sha256": sha256(args.samples_file),
            "sampling_config_file": str(
                (args.samples_file.parent / "sampling_config.json").resolve()
            ) if (args.samples_file.parent / "sampling_config.json").is_file() else None,
            "sampling_config_sha256": (
                sha256(args.samples_file.parent / "sampling_config.json")
                if (args.samples_file.parent / "sampling_config.json").is_file() else None
            ),
            "tasks_dir": str(args.tasks_dir.resolve()),
            "selected_tasks_sha256": selected_task_sha256(
                args.tasks_dir, [task["task_id"] for task in manifest["tasks"]]
            ),
            "corpus_dir": str(args.corpus_dir.resolve()),
            "corpus_index_sha256": (
                sha256(args.corpus_dir / "corpus.sqlite")
                if (args.corpus_dir / "corpus.sqlite").is_file() else None
            ),
            "git_revision": git_revision(),
            "prepare_script_sha256": sha256(Path(__file__)),
        }
        write_json_atomic(manifest_path, manifest)
        print(f"Prepared {manifest['selected_tasks']} tasks in {manifest_path}")
        print("This run tests whether same retrieved IDs preserve continuation value.")
        return

    if not manifest_path.is_file():
        parser.error(f"Missing manifest: {manifest_path}; run prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        parser.error("Unsupported manifest protocol")
    if args.mode == "run":
        if not args.model_revision:
            parser.error("run requires --model_revision for reproducibility")
        if args.repeats <= 0 or args.max_steps < 2 or args.max_tokens <= 0:
            parser.error("repeats and max_tokens must be positive; max_steps >= 2")
        if not 0 < args.temperature <= 2 or not 0 < args.top_p <= 1:
            parser.error("temperature must be in (0,2], top_p in (0,1]")
        source = manifest["source"]
        if sha256(Path(__file__)) != source["prepare_script_sha256"]:
            parser.error("Continuation pilot code changed since prepare")
        tasks_dir = Path(source["tasks_dir"])
        corpus_dir = Path(source["corpus_dir"])
        if args.tasks_dir and args.tasks_dir.resolve() != tasks_dir:
            parser.error("--tasks_dir differs from prepared manifest")
        if args.corpus_dir and args.corpus_dir.resolve() != corpus_dir:
            parser.error("--corpus_dir differs from prepared manifest")
        index = corpus_dir / "corpus.sqlite"
        if selected_task_sha256(
            tasks_dir, [task["task_id"] for task in manifest["tasks"]]
        ) != source["selected_tasks_sha256"]:
            parser.error("Selected task files changed since prepare")
        if source["corpus_index_sha256"] and sha256(index) != source["corpus_index_sha256"]:
            parser.error("Corpus index changed since prepare")
        if sha256(Path(source["samples_file"])) != source["samples_sha256"]:
            parser.error("First-search samples changed since prepare")
        sampling_config_file = source.get("sampling_config_file")
        if sampling_config_file:
            config_path = Path(sampling_config_file)
            if (not config_path.is_file()
                    or sha256(config_path) != source["sampling_config_sha256"]):
                parser.error("First-search sampling config changed since prepare")
            sampled_revision = json.loads(config_path.read_text(encoding="utf-8")).get(
                "model_revision"
            )
            if sampled_revision and sampled_revision != args.model_revision:
                parser.error("Continuation model revision differs from first-search sampling")
        wait_for_model_server(args.model_url, timeout_sec=30)
        model_ids = served_model_ids(args.model_url)
        if args.model_name not in model_ids:
            parser.error(f"Model {args.model_name!r} not served; available IDs: {model_ids}")
        run_config = {
            "protocol": PROTOCOL, "manifest_sha256": sha256(manifest_path),
            "model_url": args.model_url, "model_name": args.model_name,
            "model_revision": args.model_revision, "served_model_ids": model_ids,
            "repeats": args.repeats, "max_steps": args.max_steps,
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "history_modes": MODES,
            "git_revision": git_revision(),
            "run_script_sha256": sha256(Path(__file__)),
        }
        run_config_path = output / "run_config.json"
        if args.resume:
            if (not run_config_path.is_file()
                    or json.loads(run_config_path.read_text(encoding="utf-8")) != run_config):
                parser.error("Resume config differs from the original run")
        elif run_config_path.exists() or (output / "task_runs").exists():
            parser.error("Run already started; use --resume or a fresh output_dir")
        else:
            write_json_atomic(run_config_path, run_config)
        print("This run estimates within-/between-effect continuation reward gaps.", flush=True)
        corpus = CorpusStore(str(corpus_dir))
        corpus.load()
        actor = LLMActor(LLMClient(api_url=chat_endpoint(args.model_url),
                                   model=args.model_name))
        try:
            for index, spec in enumerate(manifest["tasks"], 1):
                result_path = output / "task_runs" / f"{spec['task_id']}.json"
                if result_path.is_file():
                    continue
                task_path = tasks_dir / f"{spec['task_id']}.json"
                if not task_path.is_file():
                    raise FileNotFoundError(task_path)
                stable_seed = manifest["selection_seed"] + int(
                    hashlib.sha256(spec["task_id"].encode()).hexdigest()[:8], 16
                )
                task_result = collect_task_continuations(
                    spec, task_path, corpus, actor,
                    repeats=args.repeats, seed=stable_seed,
                    max_steps=args.max_steps, max_tokens=args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )
                write_json_atomic(result_path, task_result)
                print(f"[{index}/{len(manifest['tasks'])}] {spec['task_id']}: "
                      f"{len(task_result['runs'])} continuations saved", flush=True)
        finally:
            corpus.close()

    task_outputs = {}
    for spec in manifest["tasks"]:
        path = output / "task_runs" / f"{spec['task_id']}.json"
        if path.is_file():
            task_outputs[spec["task_id"]] = json.loads(path.read_text(encoding="utf-8"))
    summary = analyze_continuations(manifest, task_outputs)
    summary["analysis_script_sha256"] = sha256(Path(__file__))
    write_json_atomic(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("per_task", "pairs", "limitations")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
