import json
from pathlib import Path

import pytest

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.env.rollout_collector import ConversationCollector, MULTIHOP_SYSTEM_PROMPT
from research_agent.core.env.state import EnvState
from research_agent.core.schema.document import CandidateChunk, Chunk
from research_agent.core.schema.task import TaskSample, TaskType, Rubric
from research_agent.core.tools.read import ReadTool
from research_agent.core.tools.search import SearchTool


def test_full_read_preserves_evidence_after_char_300_and_legacy_default():
    corpus = CorpusStore()
    text = "x" * 350 + " The answer is downstream."
    corpus.chunks["c"] = Chunk("c", "d", text, 0, len(text), "Title")
    state = EnvState(task=TaskSample(task_id="t", user_query="question", task_type=TaskType.SURVEY_SYNTHESIS, rubric=Rubric()), _corpus=corpus)
    state.candidate_chunks = [CandidateChunk(chunk_id="c", doc_id="d", score=1, rank=1, query="q")]
    old = ReadTool().execute({"chunk_ids": ["c"]}, state)
    assert old.data["summaries"][0]["summary"] == text[:300] + "..."
    full = ReadTool(mode="full").execute({"chunk_ids": ["c"]}, state)
    assert full.data["summaries"][0]["summary"] == text
    assert full.stats["n_truncated"] == 0
    assert state.read_summaries[0].summary == text
    with pytest.raises(ValueError, match="SEARCH"):
        ReadTool(mode="full").execute({"chunk_ids": ["unseen"]}, state)


def test_multihop_custom_prompt_is_not_overwritten():
    assert ConversationCollector(multi_hop=True).segments[0].text == MULTIHOP_SYSTEM_PROMPT
    assert ConversationCollector("custom", multi_hop=True).segments[0].text == "custom"


@pytest.mark.parametrize("params", [{"query": ""}, {"query": []}, {"query": "x", "topk": 0},
                                  {"query": "x", "topk": True}, {"query": "x", "topk": 101}])
def test_search_validates_at_execute_boundary(params):
    state = EnvState(task=TaskSample(task_id="t", user_query="q", task_type=TaskType.SURVEY_SYNTHESIS, rubric=Rubric()))
    with pytest.raises(ValueError):
        SearchTool().execute(params, state)


def test_manifest_is_stratified_deterministic_and_has_no_gold_in_public_jobs(tmp_path):
    from scripts.run_musique_protocol_smoke import prepare_manifest
    for hop in (2, 3, 4):
        for i in range(10):
            row = {"task_id": f"{hop}_{i}", "user_query": "q", "ground_truth_answer": "secret",
                   "analysis": {"hop_count": hop, "hops": [{"answer": "secret"}]}}
            (tmp_path / f"{hop}_{i}.json").write_text(json.dumps(row))
    first = prepare_manifest(tmp_path, per_hop=8, repeats=2, seed=17)
    assert first == prepare_manifest(tmp_path, per_hop=8, repeats=2, seed=17)
    assert len(first["tasks"]) == 24
    assert len(first["jobs"]) == 192
    assert "secret" not in json.dumps(first)
    assert all(sum(t["hop_count"] == hop for t in first["tasks"]) == 8 for hop in (2, 3, 4))


def test_cost_ledger_reserves_before_request_and_survives_interruption(tmp_path):
    from scripts.run_musique_protocol_smoke import CostLedger, BudgetExceeded
    ledger = CostLedger(tmp_path / "cost.jsonl", max_calls=3, max_tokens=20)
    request = ledger.reserve("job", 10)
    ledger.settle(request, prompt_tokens=20, completion_tokens=4, error=None)
    ledger.reserve("interrupted", 10)
    resumed = CostLedger(tmp_path / "cost.jsonl", max_calls=3, max_tokens=20)
    assert resumed.calls == 2
    assert resumed.charged_tokens == 14
    with pytest.raises(BudgetExceeded):
        resumed.reserve("new", 10)


def test_strict_metrics_do_not_reward_answer_contains_or_unread_citations():
    from scripts.run_musique_protocol_smoke import diagnostic_metrics
    raw = {"ground_truth_answer": "Paris", "ground_truth_answer_aliases": [],
           "ground_truth_citations": ["c"]}
    record = {"final_answer": "Paris or London", "termination_reason": "answer_submitted", "steps": []}
    assert diagnostic_metrics(record, raw)["answer_em"] == 0
    record["final_answer"] = "Paris"
    assert diagnostic_metrics(record, raw)["answer_em"] == 1
    assert diagnostic_metrics(record, raw)["grounded_em"] == 0


def test_client_records_seed_usage_and_finish_reason(monkeypatch):
    from io import BytesIO
    from research_agent.core.env.llm_client import LLMClient
    requests = []
    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"choices": [{"text": "x", "finish_reason": "length"}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode())
    monkeypatch.setattr("urllib.request.urlopen", respond)
    response = LLMClient().generate_response("q", seed=19)
    assert requests[0]["seed"] == 19
    assert response.finish_reason == "length"
    assert response.usage_reported is True
    def no_usage(request, timeout):
        return BytesIO(b'{"choices":[{"text":"x"}]}')
    monkeypatch.setattr("urllib.request.urlopen", no_usage)
    assert LLMClient().generate_response("q").usage_reported is False


def test_real_http_driver_four_arms_resume_and_manifest_guard(tmp_path):
    """A toy HTTP policy tests plumbing; it is not an experimental model result."""
    import argparse
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import sqlite3
    import threading
    from scripts.run_musique_protocol_smoke import execute

    tasks, corpus = tmp_path / "tasks", tmp_path / "corpus"
    tasks.mkdir()
    corpus.mkdir()
    with sqlite3.connect(corpus / "corpus.sqlite") as db:
        db.executescript("""
        CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT, doc_id TEXT, title TEXT, content TEXT);
        CREATE VIRTUAL TABLE chunk_fts USING fts5(title, content, content='chunks', content_rowid='rowid');
        CREATE TABLE index_metadata(key TEXT, value TEXT);
        INSERT INTO index_metadata VALUES('complete','1');
        INSERT INTO chunks VALUES(1,'c','d','Paris','Paris is the answer.');
        INSERT INTO chunk_fts(rowid,title,content) VALUES(1,'Paris','Paris is the answer.');
        """)
    for hop in (2, 3, 4):
        (tasks / f"t{hop}.json").write_text(json.dumps({
            "task_id": f"t{hop}", "user_query": "What is the answer?", "ground_truth_answer": "Paris",
            "ground_truth_citations": ["c"], "analysis": {"hop_count": hop},
            "retrieval_scope": "split_corpus"}))
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send_json(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self.send_json({"data": [{"id": "toy", "max_model_len": 32768}]})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            count = sum(m["role"] == "assistant" and bool(m["content"]) for m in body["messages"])
            action = [
                {"tool": "SEARCH", "intent": "search", "params": {"query": "Paris", "topk": 1}},
                {"tool": "READ", "intent": "read", "params": {"chunk_ids": ["c"]}},
                {"tool": "CITE", "intent": "cite", "params": {"chunk_ids": ["c"], "claims": ["Paris"]}},
                {"tool": "ANSWER", "intent": "answer", "params": {"answer_text": "Paris", "cited_chunk_ids": ["c"]}},
            ][count]
            self.send_json({"choices": [{"message": {"content": "<action>" + json.dumps(action) + "</action>"},
                                         "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 25}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = argparse.Namespace(output_dir=tmp_path / "run", tasks_dir=tasks, corpus_dir=corpus,
                              model_url=f"http://127.0.0.1:{server.server_port}/v1", model_name="toy",
                              model_dir=None, per_hop=1, repeats=1, selection_seed=17, max_steps=15,
                              max_tokens=64, temperature=0.7, top_p=0.95, prepare_only=True,
                              max_api_calls=100, max_generated_tokens=6400, max_wall_sec=60)
    try:
        assert execute(args) == 0
        assert requests == []
        args.prepare_only = False
        args.max_api_calls = 1
        assert execute(args) == 2
        assert len(requests) == 1
        assert not json.loads((args.output_dir / "summary.json").read_text())["complete"]
        args.max_api_calls = 100
        assert execute(args) == 0
        assert len(requests) == 49  # One paid partial attempt, then full resumed episodes.
        assert list((args.output_dir / "jobs").glob("*.attempt_*.json"))
        assert all(isinstance(req["seed"], int) for req in requests)
        assert all("ground_truth" not in json.dumps(req) for req in requests)
        summary = json.loads((args.output_dir / "summary.json").read_text())
        assert summary["complete"]
        assert summary["cost_accounting"]["request_reservations"] == 49
        assert summary["paired_to_cite_first_head300"]["adaptive_full"]["paired_tasks"] == 3
        assert summary["arms"]["adaptive_full"]["by_hop"]["4"]["answer_em"] == 1
        assert all(arm["diagnostic_metrics"]["grounded_em"] == 1 for arm in summary["arms"].values())
        assert execute(args) == 0
        assert len(requests) == 49  # Only completed jobs skipped, no repeat API calls.
        assert list(args.output_dir.glob("invocation_*.json"))
        args.max_tokens = 65
        with pytest.raises(ValueError, match="Manifest"):
            execute(args)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
