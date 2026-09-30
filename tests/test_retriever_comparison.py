"""Offline comparison plumbing; fixture values are not experimental results."""
import copy
import json
from pathlib import Path
import runpy

import pytest


def comparator():
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_musique_retrievers.py"
    assert path.is_file(), "Missing paired retriever comparison implementation"
    return runpy.run_path(str(path))


def fixture_runs(root):
    jobs = [{"id": f"t0_r{r}_{arm}", "task_id": "t0", "arm": arm,
             "repeat": r, "request_seed": r + 11}
            for r in (0, 1) for arm in ("cite_first_head300", "adaptive_full")]
    base = {"version": "musique-protocol-2x2-v1", "seed": 17, "per_hop": 1,
            "repeats": 2, "tasks": [{"task_id": "t0", "sha256": "task", "hop_count": 2}],
            "jobs": jobs, "config": {
                "model_name": "frozen", "model_files": {"model.safetensors": {"sha256": "weights"}},
                "code_sha256": "code", "corpus_sha256": "corpus", "max_steps": 15,
                "max_tokens": 512, "temperature": 0.7, "top_p": 0.95,
                "retrieval": {"mode": "bm25"}}}
    for name in ("bm25", "qwen_hybrid"):
        out = root / name
        (out / "jobs").mkdir(parents=True)
        manifest = copy.deepcopy(base)
        if name == "qwen_hybrid":
            manifest["config"]["retrieval"] = {
                "mode": "hybrid", "index_manifest": {"encoder_model": "Qwen3-Embedding-0.6B",
                "complete": True, "corpus_sha256": "corpus"}}
        (out / "manifest.json").write_text(json.dumps(manifest))
        for job in jobs:
            em = int(name == "qwen_hybrid" and job["repeat"] == 0)
            record = {"task_id": "t0", "job": job, "termination_reason": "answer_submitted",
                      "hop_count": 2, "episode_wall_sec": 3.0,
                      "diagnostic_metrics": {"answer_em": em, "normalized_answer_f1": em,
                        "grounded_em": em, "retrieved_gold_recall": .5, "read_gold_recall": .5,
                        "search_calls": 2, "read_calls": 1, "distinct_search_queries": 2,
                        "generation_length_finishes": 0},
                      "eval_metrics": {"citation_f1": .5, "action_parse_success_rate": 1,
                        "invalid_action_rate": 0, "average_steps": 5},
                      "token_usage": {"prompt_tokens": 50, "completion_tokens": 10}}
            (out / "jobs" / (job["id"] + ".json")).write_text(json.dumps(record))
    return jobs


def test_same_jobs_and_seeds_are_paired_without_treating_repeats_as_tasks(tmp_path):
    fixture_runs(tmp_path)
    result = comparator()["build_comparison"](tmp_path)
    assert result["complete"]
    arm = result["arms"]["adaptive_full"]
    assert arm["paired_episodes"] == 2
    assert arm["paired_tasks"] == 1
    assert arm["task_mean_deltas"]["answer_em"] == .5
    assert arm["qwen_hybrid"]["answer_em"] == .5
    assert result["costs"]["bm25"]["policy"]["unknown_or_pending_requests"] == 0


@pytest.mark.parametrize("mutation", ["task", "seed", "sampling", "model", "corpus"])
def test_changed_pair_identity_is_rejected(tmp_path, mutation):
    fixture_runs(tmp_path)
    path = tmp_path / "qwen_hybrid/manifest.json"
    m = json.loads(path.read_text())
    if mutation == "task":
        m["tasks"][0]["sha256"] = "changed"
    elif mutation == "seed":
        m["jobs"][0]["request_seed"] += 1
    elif mutation == "sampling":
        m["config"]["temperature"] = 0
    elif mutation == "model":
        m["config"]["model_files"]["model.safetensors"]["sha256"] = "changed"
    else:
        m["config"]["retrieval"]["index_manifest"]["corpus_sha256"] = "changed"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Mismatch|mismatch"):
        comparator()["build_comparison"](tmp_path)


def test_infrastructure_failure_is_not_scored_as_policy_failure(tmp_path):
    jobs = fixture_runs(tmp_path)
    path = tmp_path / "qwen_hybrid/jobs" / (jobs[0]["id"] + ".json")
    record = json.loads(path.read_text())
    record["termination_reason"] = "infrastructure_error"
    path.write_text(json.dumps(record))
    result = comparator()["build_comparison"](tmp_path)
    assert not result["complete"]
    arm = result["arms"]["cite_first_head300"]
    assert arm["paired_episodes"] == 1
    assert arm["missing_or_incomplete_jobs"]["qwen_hybrid"] == [jobs[0]["id"]]


def test_pending_requests_are_not_reported_as_free(tmp_path):
    fixture_runs(tmp_path)
    (tmp_path / "qwen_hybrid/cost.jsonl").write_text(json.dumps({
        "event": "reserve", "request": "p1", "max_tokens": 512}) + "\n")
    (tmp_path / "qwen_hybrid/embedding_cost.jsonl").write_text(json.dumps({
        "event": "start", "request": "e1", "items": 1}) + "\n")
    result = comparator()["build_comparison"](tmp_path)
    cost = result["costs"]["qwen_hybrid"]
    assert cost["policy"]["charged_completion_tokens"] == 512
    assert cost["policy"]["unknown_or_pending_requests"] == 1
    assert cost["embedding"]["unknown_or_pending_requests"] == 1
    assert not cost["embedding"]["usage_complete"]


def test_record_job_must_match_manifest(tmp_path):
    jobs = fixture_runs(tmp_path)
    path = tmp_path / "qwen_hybrid/jobs" / (jobs[0]["id"] + ".json")
    record = json.loads(path.read_text())
    record["job"]["request_seed"] = -1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Mismatch|mismatch"):
        comparator()["build_comparison"](tmp_path)


def test_failed_embedding_time_is_retained(tmp_path):
    fixture_runs(tmp_path)
    events = [{"event": "start", "request": "e1", "items": 1},
              {"event": "error", "request": "e1", "error": "timeout", "latency_sec": 7.0}]
    (tmp_path / "qwen_hybrid/embedding_cost.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events))
    cost = comparator()["build_comparison"](tmp_path)["costs"]["qwen_hybrid"]["embedding"]
    assert cost["error_requests"] == 1
    assert cost["unknown_or_pending_requests"] == 1
    assert cost["reported_client_latency_sec"] == 7.0


def test_real_prepare_manifests_compare_before_any_api_calls(tmp_path):
    import argparse
    import sqlite3
    from scripts.run_musique_protocol_smoke import execute, digest

    tasks, corpus, model, index = (tmp_path / name for name in ("tasks", "corpus", "model", "index"))
    for path in (tasks, corpus, model, index):
        path.mkdir()
    with sqlite3.connect(corpus / "corpus.sqlite") as db:
        db.execute("CREATE TABLE fixture(value TEXT)")
    (model / "config.json").write_text("{}")
    for hop in (2, 3, 4):
        (tasks / f"t{hop}.json").write_text(json.dumps({"task_id": f"t{hop}", "analysis": {"hop_count": hop}}))
    (index / "manifest.json").write_text(json.dumps({"encoder_model": "Qwen3-Embedding-0.6B",
        "complete": True, "corpus_sha256": digest(corpus / "corpus.sqlite")}))
    for number in (1, 2):
        root = tmp_path / f"compare{number}"
        for name, mode in (("bm25", "bm25"), ("qwen_hybrid", "hybrid")):
            args = argparse.Namespace(output_dir=root / name, tasks_dir=tasks, corpus_dir=corpus,
                model_url="http://127.0.0.1:1/v1", model_name="frozen", model_dir=model,
                per_hop=1, repeats=1, selection_seed=17, max_steps=15, max_tokens=512,
                temperature=.7, top_p=.95, prepare_only=True, retrieval=mode,
                dense_index=index, embedding_url="http://127.0.0.1:1/v1")
            assert execute(args) == 0  # Port1 has no server; prepare must not call it.
        comparator()["validate_manifests"](root)
        assert not comparator()["build_comparison"](root)["complete"]


def test_real_preflight_checks_index_checksum_and_never_generates(tmp_path):
    module = comparator()
    assert "preflight" in module, "Missing readiness/index preflight before either policy run"
    import hashlib
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import sqlite3
    import threading
    import numpy as np

    fixture_runs(tmp_path)
    corpus, model, embed, index = (tmp_path / name for name in ("corpus", "model", "embed", "index"))
    for path in (corpus, model, embed, index):
        path.mkdir()
    with sqlite3.connect(corpus / "corpus.sqlite") as db:
        db.executescript("""
        CREATE TABLE chunks(rowid INTEGER PRIMARY KEY,chunk_id TEXT,doc_id TEXT,title TEXT,content TEXT);
        CREATE VIRTUAL TABLE chunk_fts USING fts5(title,content,content='chunks',content_rowid='rowid');
        CREATE TABLE index_metadata(key TEXT,value TEXT);
        INSERT INTO index_metadata VALUES('complete','1');
        INSERT INTO chunks VALUES(1,'c','d','title','text');
        INSERT INTO chunk_fts VALUES('title','text');
        """)
    (model / "config.json").write_text("{}")
    (embed / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test identity only; not real model weights")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    posted = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            name, root, cap = ("frozen", model, 32768) if self.path.startswith("/policy") else (
                "Qwen3-Embedding-0.6B", embed, 4096)
            body = json.dumps({"data": [{"id": name, "root": str(root), "max_model_len": cap}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            posted.append(self.path)
            self.send_error(500, "Preflight must not generate or embed")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    np.save(index / "vectors.npy", np.array([[1., 0.]], dtype=np.float32))
    (index / "rows.json").write_text(json.dumps({"rows": [{"chunk_id": "c", "doc_id": "d"}]}))
    index_manifest = {"complete": True, "encoder_model": "Qwen3-Embedding-0.6B",
        "corpus_sha256": sha(corpus / "corpus.sqlite"), "input_policy": "local_hf_token_ids_v1",
        "query_instruction": "", "server": {"id": "Qwen3-Embedding-0.6B", "root": str(embed),
        "max_model_len": 4096}, "encoder_fingerprint": {"root": str(embed), "pooling": "LAST",
        "files": {"config.json": sha(embed / "config.json")}}, "dimension": 2,
        "vectors_sha256": sha(index / "vectors.npy"), "rows_file_sha256": sha(index / "rows.json")}
    (index / "manifest.json").write_text(json.dumps(index_manifest))
    for name in ("bm25", "qwen_hybrid"):
        path = tmp_path / name / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config"].update({"model_dir": str(model), "corpus_dir": str(corpus),
            "corpus_sha256": index_manifest["corpus_sha256"],
            "model_files": {"model.safetensors": {"sha256": sha(model / "model.safetensors")}},
            "model_url": f"http://127.0.0.1:{server.server_port}/policy/v1"})
        if name == "qwen_hybrid":
            manifest["config"]["retrieval"].update({"index_dir": str(index),
                "index_manifest": index_manifest, "embedding_url": f"http://127.0.0.1:{server.server_port}/embed/v1"})
        path.write_text(json.dumps(manifest))
    try:
        module["preflight"](tmp_path)
        assert posted == []
        (index / "vectors.npy").write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="checksum"):
            module["preflight"](tmp_path)
        assert posted == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
