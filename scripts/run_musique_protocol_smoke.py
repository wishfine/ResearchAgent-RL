#!/usr/bin/env python3
"""Frozen-policy 2x2 protocol diagnostic, not training or a benchmark claim.

Gold annotations are used only for sampling and offline metrics. API requests
contain the public question, system protocol and actual tool observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_agent.baselines.llm_actor import LLMActor
from research_agent.core.baseline_runner import aggregate_records, run_episode
from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.llm_client import LLMClient
from research_agent.core.env.rollout_collector import CITE_FIRST_SYSTEM_PROMPT, ADAPTIVE_SYSTEM_PROMPT
from research_agent.core.tools.read import ReadTool
from scripts.run_llm_baseline import chat_endpoint, load_task, make_env, models_endpoint

VERSION = "musique-protocol-2x2-v1"
INCOMPLETE = {"actor_error", "infrastructure_error"}
ARMS = {
    "cite_first_head300": (CITE_FIRST_SYSTEM_PROMPT, "head300"),
    "cite_first_full": (CITE_FIRST_SYSTEM_PROMPT, "full"),
    "adaptive_head300": (ADAPTIVE_SYSTEM_PROMPT, "head300"),
    "adaptive_full": (ADAPTIVE_SYSTEM_PROMPT, "full"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_manifest(tasks_dir: Path, per_hop: int, repeats: int, seed: int) -> dict:
    if per_hop < 1 or repeats < 1:
        raise ValueError("per_hop and repeats must be positive")
    buckets = {hop: [] for hop in (2, 3, 4)}
    for path in sorted(tasks_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        hop = raw.get("analysis", {}).get("hop_count")
        if hop in buckets:
            buckets[hop].append({"task_id": raw["task_id"], "file": path.name,
                                 "hop_count": hop, "sha256": digest(path)})
    rng = random.Random(seed)
    tasks = []
    for hop, bucket in buckets.items():
        if len(bucket) < per_hop:
            raise ValueError(f"Need {per_hop} tasks at hop {hop}, got {len(bucket)}")
        rng.shuffle(bucket)
        tasks.extend(bucket[:per_hop])
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    jobs = []
    for index, task in enumerate(tasks):
        for repeat in range(repeats):
            for arm in ARMS:
                job_id = f"t{index:03d}_r{repeat:02d}_{arm}"
                jobs.append({"id": job_id, "task_id": task["task_id"], "arm": arm,
                             "repeat": repeat, "request_seed": int.from_bytes(
                                 hashlib.sha256(f"{seed}:{job_id}".encode()).digest()[:4], "big") % 2**31})
    rng.shuffle(jobs)
    return {"version": VERSION, "seed": seed, "per_hop": per_hop,
            "repeats": repeats, "tasks": tasks, "jobs": jobs}


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    """Write-ahead reservations: unknown/failed/pending calls are NOT free."""
    def __init__(self, path: Path, max_calls: int, max_tokens: int):
        self.path, self.max_calls, self.max_tokens = path, max_calls, max_tokens
        self.pending = {}
        self.calls = 0
        self.charged_tokens = 0
        self.settled = set()
        if path.exists():
            for line in path.read_text().splitlines():
                self._apply(json.loads(line))

    def _apply(self, event):
        request = event["request"]
        if event["event"] == "reserve":
            if request in self.pending:
                raise ValueError("Duplicate request reservation")
            self.pending[request] = event["max_tokens"]
            self.calls += 1
            self.charged_tokens += event["max_tokens"]
        elif event["event"] == "settle":
            if request not in self.pending or request in self.settled:
                raise ValueError("Invalid ledger settlement")
            self.settled.add(request)
            actual = event["completion_tokens"]
            if actual is not None:
                self.charged_tokens += actual - self.pending[request]
        else:
            raise ValueError("Unknown cost event")

    def _append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(event)

    def reserve(self, job: str, max_tokens: int) -> str:
        if self.calls >= self.max_calls or self.charged_tokens + max_tokens > self.max_tokens:
            raise BudgetExceeded("API-call or completion-token budget exhausted")
        request = f"request_{self.calls:06d}"
        self._append({"event": "reserve", "request": request, "job": job,
                      "max_tokens": max_tokens, "time": time.time()})
        return request

    def settle(self, request, *, prompt_tokens, completion_tokens, error, **metadata):
        self._append({"event": "settle", "request": request, "prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens, "error": error,
                      "time": time.time(), **metadata})


class BudgetActor:
    def __init__(self, actor, ledger, job, deadline):
        self.actor, self.ledger, self.job, self.deadline = actor, ledger, job, deadline

    def decide(self, prompt, **kwargs):
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("Wall-clock budget exhausted before request")
        request = self.ledger.reserve(self.job, kwargs["max_tokens"])
        started = time.monotonic()
        try:
            turn = self.actor.decide(prompt, **kwargs)
        except Exception as exc:
            self.ledger.settle(request, prompt_tokens=None, completion_tokens=None,
                               error=str(exc), latency_sec=time.monotonic() - started,
                               seed=kwargs.get("seed"))
            raise
        self.ledger.settle(request, prompt_tokens=turn.prompt_tokens if turn.usage_reported else None,
                           completion_tokens=turn.completion_tokens if turn.usage_reported else None,
                           error=None, latency_sec=turn.latency_sec, seed=kwargs.get("seed"),
                           finish_reason=turn.finish_reason,
                           prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        return turn


def normalize(text):
    import string
    text = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def diagnostic_metrics(record: dict, raw: dict) -> dict:
    prediction = normalize(record["final_answer"])
    references = [normalize(answer) for answer in
                  [raw.get("ground_truth_answer", ""), *raw.get("ground_truth_answer_aliases", [])]]
    em = int(bool(prediction) and prediction in references)
    def f1(reference):
        a, b = prediction.split(), reference.split()
        overlap = sum((Counter(a) & Counter(b)).values())
        return 2 * overlap / (len(a) + len(b)) if a and b else 0.0
    retrieved, read, cited, answer_cites = set(), set(), set(), set()
    counts = Counter()
    search_queries = []
    truncated = 0
    for step in record["steps"]:
        action, result = step.get("action") or {}, step.get("tool_result") or {}
        counts[action.get("tool", "INVALID")] += 1
        truncated += int(step.get("finish_reason") == "length")
        if not result.get("success") or step.get("error"):
            continue
        data = result.get("data") or {}
        tool = action.get("tool")
        if tool == "SEARCH":
            retrieved.update(c["chunk_id"] for c in data.get("candidates", []))
            search_queries.append(action.get("params", {}).get("query", ""))
        elif tool == "READ":
            read.update(c["chunk_id"] for c in data.get("summaries", []))
        elif tool == "CITE":
            cited.update(data.get("chunk_ids", []))
        elif tool == "ANSWER":
            answer_cites = set(data.get("cited_chunk_ids", []))
    gold = set(raw.get("ground_truth_citations", []))
    grounded = bool(em and gold and gold <= answer_cites and answer_cites <= read & cited
                    and record["termination_reason"] == "answer_submitted")
    return {"answer_em": em, "normalized_answer_f1": max(map(f1, references)),
            "grounded_em": int(grounded),
            "retrieved_gold_recall": len(gold & retrieved) / len(gold) if gold else None,
            "read_gold_recall": len(gold & read) / len(gold) if gold else None,
            "search_calls": counts["SEARCH"], "read_calls": counts["READ"],
            "distinct_search_queries": len(set(search_queries)),
            "generation_length_finishes": truncated}


def summarize(out: Path, manifest: dict):
    arms = {}
    all_valid = {}
    expected = len(manifest["tasks"]) * manifest["repeats"]
    for arm in ARMS:
        records = []
        for job in manifest["jobs"]:
            path = out / "jobs" / (job["id"] + ".json")
            if job["arm"] == arm and path.exists():
                records.append(json.loads(path.read_text()))
        valid = [r for r in records if r["termination_reason"] not in INCOMPLETE]
        all_valid[arm] = {(r["task_id"], r["job"]["repeat"]): r for r in valid}
        metrics = aggregate_records(valid)
        diagnostic = {}
        if valid:
            for key in valid[0]["diagnostic_metrics"]:
                values = [r["diagnostic_metrics"][key] for r in valid if r["diagnostic_metrics"][key] is not None]
                diagnostic[key] = sum(values) / len(values) if values else None
        archived = [json.loads(path.read_text()) for path in (out / "jobs").glob("*.attempt_*.json")]
        arms[arm] = {"expected_episodes": expected, "completed_episodes": len(valid),
                     "interrupted_attempts_saved": len(records) - len(valid) +
                         sum(r["job"]["arm"] == arm for r in archived),
                     "legacy_metrics": metrics, "diagnostic_metrics": diagnostic,
                     "by_hop": {str(hop): {
                         "completed_episodes": sum(r["hop_count"] == hop for r in valid),
                         "answer_em": sum(r["diagnostic_metrics"]["answer_em"] for r in valid if r["hop_count"] == hop)
                                      / sum(r["hop_count"] == hop for r in valid)
                                      if any(r["hop_count"] == hop for r in valid) else None,
                     } for hop in (2, 3, 4)},
                     "complete": len(valid) == expected}
    baseline = all_valid["cite_first_head300"]
    paired = {}
    for arm, results in all_valid.items():
        common = sorted(set(results) & set(baseline))
        paired[arm] = {"paired_episodes": len(common), "paired_tasks": len({key[0] for key in common}),
                       "answer_em_delta": sum(results[key]["diagnostic_metrics"]["answer_em"] -
                                              baseline[key]["diagnostic_metrics"]["answer_em"] for key in common)
                                          / len(common) if common else None,
                       "note": "Descriptive means; repeats are not independent tasks."}
    cost_path = out / "cost.jsonl"
    costs = [json.loads(line) for line in cost_path.read_text().splitlines()] if cost_path.exists() else []
    settled = [event for event in costs if event["event"] == "settle"]
    reservations = [event for event in costs if event["event"] == "reserve"]
    accounted = {event["request"]: event["completion_tokens"] for event in settled
                 if event["completion_tokens"] is not None}
    cost_summary = {"request_reservations": len(reservations),
                    "unknown_or_failed_or_pending_requests": len(reservations) - len(accounted),
                    "reported_prompt_tokens": sum(event["prompt_tokens"] or 0 for event in settled),
                    "reported_completion_tokens": sum(accounted.values()),
                    "charged_completion_tokens": sum(accounted.get(event["request"], event["max_tokens"]) for event in reservations)}
    result = {"version": VERSION, "complete": all(a["complete"] for a in arms.values()),
              "note": "Exploratory balanced-hop train probe, not held-out benchmark performance.",
              "arms": arms, "paired_to_cite_first_head300": paired, "cost_accounting": cost_summary}
    atomic_json(out / "summary.json", result)
    return result


def execute(args) -> int:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Unix lock prevents two drivers corrupting a shared run directory.
    lock = (out / "driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks_dir, corpus_dir = args.tasks_dir.resolve(), args.corpus_dir.resolve()
    manifest = prepare_manifest(tasks_dir, args.per_hop, args.repeats, args.selection_seed)
    corpus_path = corpus_dir / "corpus.sqlite"
    if not corpus_path.is_file():
        raise ValueError("This diagnostic requires the MuSiQue SQLite corpus")
    code_paths = [Path(__file__).resolve(), ROOT / "scripts/run_llm_baseline.py"]
    code_paths.extend(sorted((ROOT / "research_agent").rglob("*.py")))
    code_hash = hashlib.sha256("".join(str(p.relative_to(ROOT)) + digest(p) for p in code_paths).encode()).hexdigest()
    manifest["config"] = {"model_url": args.model_url, "model_name": args.model_name,
                          "model_dir": str(args.model_dir.resolve()) if args.model_dir else None,
                          "tasks_dir": str(tasks_dir), "corpus_dir": str(corpus_dir),
                          "corpus_sha256": digest(corpus_path), "code_sha256": code_hash,
                          "max_steps": args.max_steps, "max_tokens": args.max_tokens,
                          "temperature": args.temperature, "top_p": args.top_p,
                          "prompts": {name: {"text": prompt, "read_mode": mode} for name, (prompt, mode) in ARMS.items()}}
    retrieval = getattr(args, "retrieval", "bm25")
    index_path = getattr(args, "dense_index", None)
    if retrieval != "bm25":
        if index_path is None:
            raise ValueError("Dense/hybrid mode requires --dense_index")
        index_manifest = json.loads((index_path / "manifest.json").read_text())
        manifest["config"]["retrieval"] = {"mode": retrieval, "index_dir": str(index_path.resolve()),
                                          "index_manifest": index_manifest,
                                          "embedding_url": args.embedding_url,
                                          "candidate_pool": 50, "rrf_k": 60}
    else:
        manifest["config"]["retrieval"] = {"mode": "bm25"}
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest["config"]["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    if args.model_dir:
        if not args.model_dir.is_dir() or not (args.model_dir / "config.json").is_file():
            raise ValueError("model_dir must be a materialized model with config.json")
        manifest["config"]["model_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(args.model_dir.glob("*")) if p.is_file() and
            (p.suffix in {".json", ".safetensors", ".jinja"})}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Manifest/config/code/data changed. Use a new output directory.")
    else:
        atomic_json(manifest_path, manifest)
    print(f"Prepared {len(manifest['tasks'])} tasks / {len(manifest['jobs'])} episodes in {out}", flush=True)
    if args.prepare_only:
        return 0
    with urllib.request.urlopen(models_endpoint(args.model_url), timeout=10) as response:
        api_models = json.load(response)
    matching = [model for model in api_models.get("data", []) if model.get("id") == args.model_name]
    if len(matching) != 1:
        raise ValueError("Served model ID does not match requested model")
    if args.model_dir and matching[0].get("root") != str(args.model_dir.resolve()):
        raise ValueError("Served model root does not match frozen model directory")
    atomic_json(out / "server_models.json", api_models)
    if matching[0].get("max_model_len") != 32768:
        raise ValueError("This diagnostic requires an explicitly reported 32768-context model server")
    atomic_json(out / f"invocation_{time.time_ns()}.json", {
        "max_api_calls": args.max_api_calls, "max_generated_tokens": args.max_generated_tokens,
        "max_wall_sec": args.max_wall_sec, "start_time": time.time(),
        "python": sys.version, "executable": sys.executable})
    ledger = CostLedger(out / "cost.jsonl", args.max_api_calls, args.max_generated_tokens)
    corpus = CorpusStore(str(corpus_dir))
    corpus.load()
    if retrieval != "bm25":
        from research_agent.core.corpus.dense import DenseCorpus, EmbeddingClient
        encoder = EmbeddingClient(args.embedding_url, index_manifest["encoder_model"],
                                  index_manifest["query_instruction"], event_log=out / "embedding_cost.jsonl",
                                  fingerprint=index_manifest["encoder_fingerprint"])
        corpus = DenseCorpus(corpus, index_path, encoder, retrieval,
                             corpus_sha256=manifest["config"]["corpus_sha256"])
    actor = LLMActor(LLMClient(api_url=chat_endpoint(args.model_url), model=args.model_name))
    by_id = {row["task_id"]: row for row in manifest["tasks"]}
    deadline = time.monotonic() + args.max_wall_sec
    try:
        for index, job in enumerate(manifest["jobs"], 1):
            output = out / "jobs" / (job["id"] + ".json")
            if output.exists() and json.loads(output.read_text())["termination_reason"] not in INCOMPLETE:
                continue
            task_path = tasks_dir / by_id[job["task_id"]]["file"]
            if digest(task_path) != by_id[job["task_id"]]["sha256"]:
                raise ValueError("Task file changed during run")
            raw = json.loads(task_path.read_text())
            if raw.get("retrieval_scope") != "split_corpus" or raw.get("reference_docs"):
                raise ValueError("Protocol diagnostic requires pooled split_corpus tasks without reference_docs")
            task = load_task(str(task_path))
            prompt, mode = ARMS[job["arm"]]
            env = make_env(corpus, args.max_steps)
            env.register_tool(ReadTool(mode=mode))
            started = time.monotonic()
            record = run_episode(env, task, BudgetActor(actor, ledger, job["id"], deadline),
                                 max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                                 system_prompt=prompt, request_seed_base=job["request_seed"])
            record.update({"job": job, "hop_count": raw["analysis"]["hop_count"],
                           "episode_wall_sec": time.monotonic() - started,
                           "diagnostic_metrics": diagnostic_metrics(record, raw)})
            if output.exists():
                # Preserve an unsuccessful previous attempt, including its partial trajectory.
                os.replace(output, output.with_name(output.stem + f".attempt_{time.time_ns()}.json"))
            atomic_json(output, record)
            summary = summarize(out, manifest)
            print(f"[{index}/{len(manifest['jobs'])}] {job['id']} {record['termination_reason']} "
                  f"EM={record['diagnostic_metrics']['answer_em']} calls={ledger.calls}", flush=True)
            if record["termination_reason"] in INCOMPLETE:
                print(record["steps"][-1]["error"], flush=True)
                return 2
        summary = summarize(out, manifest)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["complete"] else 2
    finally:
        corpus.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks_dir", type=Path, required=True)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_url", default="http://127.0.0.1:8105/v1")
    parser.add_argument("--model_name", default="Qwen3.5-9B-SFT874")
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--per_hop", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--selection_seed", type=int, default=20260929)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_api_calls", type=int, default=2880)
    parser.add_argument("--max_generated_tokens", type=int, default=1474560)
    parser.add_argument("--max_wall_sec", type=float, default=14400)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--retrieval", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--dense_index", type=Path)
    parser.add_argument("--embedding_url", default="http://127.0.0.1:8106/v1")
    args = parser.parse_args()
    if any(value <= 0 for value in (args.max_steps, args.max_tokens, args.max_api_calls,
                                   args.max_generated_tokens, args.max_wall_sec)):
        parser.error("All budgets must be positive")
    try:
        return execute(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
