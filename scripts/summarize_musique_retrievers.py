#!/usr/bin/env python3
"""Offline, identity-checked BM25 vs Qwen hybrid frozen-policy comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import urllib.request

NAMES = ("bm25", "qwen_hybrid")
INCOMPLETE = {"actor_error", "infrastructure_error"}
DIAGNOSTIC = ("answer_em", "normalized_answer_f1", "grounded_em", "retrieved_gold_recall",
              "read_gold_recall", "search_calls", "read_calls", "distinct_search_queries",
              "generation_length_finishes")
LEGACY = ("citation_f1", "action_parse_success_rate", "invalid_action_rate", "average_steps")


def load(path):
    return json.loads(Path(path).read_text())


def validate_manifests(root):
    manifests = {name: load(root / name / "manifest.json") for name in NAMES}
    common = []
    for name, manifest in manifests.items():
        value = {**manifest, "config": dict(manifest["config"])}
        retrieval = value["config"].pop("retrieval")
        if retrieval.get("mode") != ("bm25" if name == "bm25" else "hybrid"):
            raise ValueError(f"Retrieval mismatch: {name}")
        if name == "qwen_hybrid":
            index = retrieval["index_manifest"]
            if (index.get("encoder_model") != "Qwen3-Embedding-0.6B" or
                    not index.get("complete") or
                    index.get("corpus_sha256") != value["config"]["corpus_sha256"]):
                raise ValueError("Qwen index/corpus mismatch")
        if not value["config"].get("model_files"):
            raise ValueError("Missing frozen model file identity")
        ids = [job["id"] for job in manifest["jobs"]]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Duplicate/empty jobs")
        common.append(value)
    if common[0] != common[1]:
        raise ValueError("Mismatch in tasks/seeds/model/code/corpus/sampling/prompts across retrievers")
    # Compare stable server identity if both completed clients saved it.
    server_files = [root / name / "server_models.json" for name in NAMES]
    if all(path.is_file() for path in server_files):
        identities = []
        for path in server_files:
            models = load(path).get("data", [])
            match = [m for m in models if m.get("id") == common[0]["config"]["model_name"]]
            if len(match) != 1:
                raise ValueError("Server model mismatch")
            identities.append(tuple(match[0].get(k) for k in ("id", "root", "max_model_len")))
        if identities[0] != identities[1]:
            raise ValueError("Server identity mismatch")
    return manifests, hashlib.sha256(json.dumps(common[0], sort_keys=True).encode()).hexdigest()


def costs(out, embedding=False):
    path = out / ("embedding_cost.jsonl" if embedding else "cost.jsonl")
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    start_kind, end_kind = ("start", "success") if embedding else ("reserve", "settle")
    starts, ends, errors = {}, {}, set()
    for event in events:
        request = event["request"]
        if event["event"] == start_kind:
            if request in starts:
                raise ValueError(f"Duplicate cost reservation in {path}")
            starts[request] = event
        elif event["event"] == end_kind:
            if request not in starts or request in ends:
                raise ValueError(f"Invalid cost settlement in {path}")
            ends[request] = event
            if event.get("error"):
                errors.add(request)
        elif embedding and event["event"] == "error":
            if request not in starts or request in errors or request in ends:
                raise ValueError(f"Invalid embedding error in {path}")
            errors.add(request)
        else:
            raise ValueError(f"Unknown cost event in {path}")
    known = {key: event for key, event in ends.items() if
             (isinstance(event.get("usage"), dict) and event["usage"].get("prompt_tokens") is not None
              if embedding else event.get("completion_tokens") is not None)}
    result = {"ledger_present": path.exists(), "request_reservations": len(starts),
              "unknown_or_pending_requests": len(starts) - len(known),
              "error_requests": len(errors), "usage_complete": path.exists() and len(known) == len(starts)}
    if embedding:
        result["reported_prompt_tokens"] = sum(e["usage"]["prompt_tokens"] for e in known.values())
        result["reported_client_latency_sec"] = sum(e.get("latency_sec", 0) for e in events
                                                    if e["event"] in {"success", "error"})
    else:
        result["reported_prompt_tokens"] = sum(e.get("prompt_tokens") or 0 for e in ends.values())
        result["reported_completion_tokens"] = sum(e["completion_tokens"] for e in known.values())
        result["charged_completion_tokens"] = sum(
            known[k]["completion_tokens"] if k in known else e["max_tokens"] for k, e in starts.items()
        ) if path.exists() else None
    return result


def preflight(root):
    """Read-only full index and /models checks; no generation/embedding POST."""
    root = Path(root)
    manifests, identity = validate_manifests(root)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_llm_baseline import models_endpoint
    from research_agent.core.corpus.store import CorpusStore
    from research_agent.core.corpus.dense import DenseCorpus, EmbeddingClient, file_hash
    config = manifests["qwen_hybrid"]["config"]
    with urllib.request.urlopen(models_endpoint(config["model_url"]), timeout=10) as response:
        served = json.load(response)
    models = [model for model in served.get("data", []) if model.get("id") == config["model_name"]]
    if len(models) != 1 or (models[0].get("root") != str(Path(config["model_dir"]).resolve()) or
                            models[0].get("max_model_len") != 32768):
        raise ValueError("Policy service model/root/32768-context mismatch")
    for name, expected in config["model_files"].items():
        if file_hash(Path(config["model_dir"]) / name) != expected["sha256"]:
            raise ValueError("Frozen policy file mismatch")
    corpus_path = Path(config["corpus_dir"]) / "corpus.sqlite"
    if file_hash(corpus_path) != config["corpus_sha256"]:
        raise ValueError("Live corpus mismatch")
    retrieval = config["retrieval"]
    manifest = retrieval["index_manifest"]
    encoder = EmbeddingClient(retrieval["embedding_url"], manifest["encoder_model"],
                              manifest["query_instruction"], fingerprint=manifest["encoder_fingerprint"])
    corpus = CorpusStore(config["corpus_dir"])
    try:
        corpus.load()
        dense = DenseCorpus(corpus, retrieval["index_dir"], encoder, "hybrid",
                            corpus_sha256=config["corpus_sha256"])
        if dense.manifest != manifest:
            raise ValueError("Live index manifest mismatch")
        return {"paired_manifest_sha256": identity, "policy_server": models[0],
                "embedding_server": manifest["server"], "index_shape": list(dense.vectors.shape),
                "checks": "Live model/corpus/index hashes and APIs checked; no inference POST."}
    finally:
        corpus.close()


def metrics(record):
    values = {k: record["diagnostic_metrics"][k] for k in DIAGNOSTIC}
    values.update({k: record["eval_metrics"][k] for k in LEGACY})
    values.update({"episode_wall_sec": record["episode_wall_sec"],
                   "prompt_tokens": record["token_usage"]["prompt_tokens"],
                   "completion_tokens": record["token_usage"]["completion_tokens"]})
    return values


def means(rows):
    return {key: statistics.mean(row[key] for row in rows)
            if all(row[key] is not None for row in rows) else None
            for key in rows[0]} if rows else {}


def build_comparison(root):
    root = Path(root)
    manifests, identity = validate_manifests(root)
    jobs = manifests["bm25"]["jobs"]
    arms, resource_costs = {}, {}
    records = {name: {} for name in NAMES}
    for name in NAMES:
        for job in jobs:
            path = root / name / "jobs" / (job["id"] + ".json")
            if not path.is_file():
                continue
            record = load(path)
            if record["job"] != job or record["task_id"] != job["task_id"]:
                raise ValueError(f"Record/job mismatch: {path}")
            if record["termination_reason"] not in INCOMPLETE:
                records[name][job["id"]] = record
        resource_costs[name] = {"policy": costs(root / name),
                               "embedding": costs(root / name, embedding=True)}
    for arm in sorted({job["arm"] for job in jobs}):
        expected = [job for job in jobs if job["arm"] == arm]
        paired = [job for job in expected if all(job["id"] in records[name] for name in NAMES)]
        # Average repeats within each task first; no independent-task claim for repeats.
        task_deltas = defaultdict(list)
        for job in paired:
            a, b = (metrics(records[name][job["id"]]) for name in NAMES)
            task_deltas[job["task_id"]].append({k: b[k] - a[k] if a[k] is not None and b[k] is not None
                                                else None for k in a})
        arms[arm] = {
            "expected_episodes_per_retriever": len(expected), "paired_episodes": len(paired),
            "paired_tasks": len(task_deltas),
            "complete": len(paired) == len(expected),
            "missing_or_incomplete_jobs": {name: [job["id"] for job in expected
                if job["id"] not in records[name]] for name in NAMES},
            **{name: means([metrics(records[name][job["id"]]) for job in paired]) for name in NAMES},
            "task_mean_deltas": means([means(rows) for rows in task_deltas.values()]),
        }
    return {"version": "musique-retriever-comparison-v1", "complete": all(a["complete"] for a in arms.values()),
            "paired_manifest_sha256": identity, "tasks": len(manifests["bm25"]["tasks"]),
            "repeats": manifests["bm25"]["repeats"], "arms": arms, "costs": resource_costs,
            "note": "Balanced-hop train discovery; descriptive task-averaged differences, not significance or RL gains. "
                    "Incomplete pairs excluded, never scored as zero. Logged costs include failed/retried attempts; "
                    "missing ledgers do not prove zero cost. No GPU/embedding build cost estimate."}


def render(result):
    lines = ["# Frozen SFT874: BM25 vs Qwen hybrid", "",
             f"Complete: {result['complete']}; tasks: {result['tasks']}; repeats: {result['repeats']}",
             "", result["note"], "", "Positive deltas favor Qwen hybrid only for quality metrics; "
             "higher token, latency, invalid-rate or call deltas mean higher cost/errors.", "",
             "| Arm | Paired episodes/tasks | BM25 EM | Qwen EM | Δ EM | Δ grounded EM | Δ wall sec |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, values in result["arms"].items():
        if not values["paired_episodes"]:
            lines.append(f"| {arm} | 0/0 | pending | pending | pending | pending | pending |")
            continue
        a, b, d = values["bm25"], values["qwen_hybrid"], values["task_mean_deltas"]
        lines.append(f"| {arm} | {values['paired_episodes']}/{values['paired_tasks']} | "
                     f"{a['answer_em']:.3f} | {b['answer_em']:.3f} | {d['answer_em']:+.3f} | "
                     f"{d['grounded_em']:+.3f} | {d['episode_wall_sec']:+.3f} |")
    lines += ["", "Full recall, citation, parse, token, action and paid-attempt cost fields are in comparison.json.",
              "Policy token fields are reported usage; missing usage must be read alongside cost ledger flags."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        evidence = preflight(args.root)
        path = args.root / "preflight.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(path)
        print("Policy/embedding services, frozen files and complete index checksums: OK (no inference POST).")
        return 0
    if args.validate_only:
        validate_manifests(args.root)
        print("Paired task/model/code/sampling identity: OK; no model calls made.")
        return 0
    result = build_comparison(args.root)
    for name, text in (("comparison.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
                       ("comparison.md", render(result))):
        target = args.root / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(target)
    print(render(result), end="")
    return 2 if args.require_complete and not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
