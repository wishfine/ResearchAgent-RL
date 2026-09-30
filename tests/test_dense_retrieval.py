import json
from pathlib import Path

import numpy as np
import pytest

from research_agent.core.corpus.store import CorpusStore
from research_agent.core.schema.document import Chunk


def toy_corpus():
    corpus = CorpusStore()
    for cid, doc, text in [("a", "d1", "lexical alpha"), ("b", "d2", "semantic beta"), ("c", "d2", "other")]:
        corpus._add_chunk(Chunk(cid, doc, text, 0, len(text), cid))
    return corpus


class ToyEncoder:
    model = "toy"
    query_instruction = "instruction"
    def encode(self, texts):
        # This is a test encoder, never a claimed research model result.
        return np.array([[1., 0.] if "beta" in text else [0., 1.] for text in texts], dtype=np.float32)
    def verify_server(self):
        return {"id": "toy", "root": "toy-snapshot"}


def test_dense_and_hybrid_are_scoped_and_keep_interface(tmp_path):
    from research_agent.core.corpus.dense import DenseCorpus, build_index
    corpus = toy_corpus()
    build_index(corpus, tmp_path, ToyEncoder(), corpus_sha256="toy-corpus", batch_size=2)
    dense = DenseCorpus(corpus, tmp_path, ToyEncoder(), "dense", corpus_sha256="toy-corpus")
    assert dense.search("beta", 1)[0].chunk_id == "b"
    assert dense.search("beta", 3, ["d1"])[0].chunk_id == "a"
    assert dense.search("beta", 3, []) == []
    hybrid = DenseCorpus(corpus, tmp_path, ToyEncoder(), "hybrid", corpus_sha256="toy-corpus")
    results = hybrid.search("alpha", 3)
    assert len({candidate.chunk_id for candidate in results}) == len(results)
    assert all(candidate.query == "alpha" for candidate in results)
    assert [candidate.rank for candidate in results] == list(range(1, len(results) + 1))
    assert hybrid.get_chunk("b").content == "semantic beta"
    assert "b" in hybrid


def test_index_resume_mismatch_and_corruption_fail_closed(tmp_path):
    from research_agent.core.corpus.dense import DenseCorpus, build_index
    corpus = toy_corpus()
    class Broken(ToyEncoder):
        count = 0
        def encode(self, texts):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("interrupted")
            return super().encode(texts)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_index(corpus, tmp_path, Broken(), corpus_sha256="source", batch_size=2)
    assert json.loads((tmp_path / "progress.json").read_text())["rows_done"] == 2
    build_index(corpus, tmp_path, ToyEncoder(), corpus_sha256="source", batch_size=2)
    with pytest.raises(ValueError, match="corpus"):
        DenseCorpus(corpus, tmp_path, ToyEncoder(), "dense", corpus_sha256="other")
    vectors = np.load(tmp_path / "vectors.npy", mmap_mode="r+")
    vectors[0] = [9., 9.]
    vectors.flush()
    with pytest.raises(ValueError, match="checksum"):
        DenseCorpus(corpus, tmp_path, ToyEncoder(), "dense", corpus_sha256="source")


def test_embedding_http_response_order_and_validation(monkeypatch):
    from io import BytesIO
    from research_agent.core.corpus.dense import EmbeddingClient
    requests = []
    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"data": [{"index": 1, "embedding": [0., 2.]},
                                              {"index": 0, "embedding": [3., 0.]}],
                                  "usage": {"prompt_tokens": 12}}).encode())
    monkeypatch.setattr("urllib.request.urlopen", respond)
    client = EmbeddingClient("http://localhost:8106/v1", "toy")
    vectors = client.encode(["one", "two"])
    assert np.allclose(vectors, [[1., 0.], [0., 1.]])
    assert requests[0]["encoding_format"] == "float"
    assert requests[0]["input"] == ["one", "two"]
    def bad(request, timeout):
        return BytesIO(b'{"data":[{"index":0,"embedding":[0,0]}]}')
    monkeypatch.setattr("urllib.request.urlopen", bad)
    with pytest.raises(Exception, match="zero|count"):
        client.encode(["q"])


def test_embedding_failure_is_infrastructure_not_policy_invalid():
    from research_agent.baselines.llm_actor import ActorTurn
    from research_agent.core.baseline_runner import run_episode
    from research_agent.core.env.env import ResearchEnv
    from research_agent.core.schema.action import Action
    from research_agent.core.schema.task import TaskSample, TaskType, Rubric
    from research_agent.core.tools.base import ToolInfrastructureError
    from research_agent.core.tools.search import SearchTool
    class FailingCorpus(CorpusStore):
        def search(self, *args, **kwargs):
            raise ToolInfrastructureError("embedding unavailable")
    class Actor:
        def decide(self, prompt, **kwargs):
            return ActorTurn("raw", Action.search("query"), "", 0.1, 20, 10)
    env = ResearchEnv(FailingCorpus(), max_steps=4)
    env.register_tool(SearchTool())
    task = TaskSample("t", TaskType.SURVEY_SYNTHESIS, "q", Rubric(), retrieval_scope="split_corpus")
    result = run_episode(env, task, Actor())
    assert result["termination_reason"] == "infrastructure_error"
    assert env._state.invalid_action_count == 0
    assert "embedding unavailable" in result["steps"][-1]["error"]


def test_windows_cover_tail_without_exceeding_encoder_limit():
    from scripts.build_dense_index import token_windows
    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2
        def encode(self, text, add_special_tokens=False):
            return list(range(1100))
        def build_inputs_with_special_tokens(self, tokens):
            return [-1] + tokens + [-2]
    windows = token_windows(Tokenizer(), "long text", max_length=512, overlap=64)
    assert all(len(window) <= 512 for window in windows)
    assert set(token for window in windows for token in window[1:-1]) == set(range(1100))
    assert windows[-1][-2] == 1099
    with pytest.raises(ValueError):
        token_windows(Tokenizer(), "q", max_length=64, overlap=64)


@pytest.mark.parametrize("model_type,pooling", [("bert", "CLS"), ("qwen3", "LAST")])
def test_model_fingerprint_uses_model_specific_pooling(tmp_path, model_type, pooling):
    from scripts.build_dense_index import model_fingerprint
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "model.safetensors").write_bytes(b"test weight identity")
    result = model_fingerprint(tmp_path)
    assert result["pooling"] == pooling
    assert result["normalize"] is True
    with pytest.raises(ValueError):
        model_fingerprint(tmp_path, "LAST" if pooling == "CLS" else "CLS")


def test_multiple_windows_collapse_to_original_paragraph_id(tmp_path):
    from research_agent.core.corpus.dense import DenseCorpus, build_index
    corpus = toy_corpus()
    build_index(corpus, tmp_path, ToyEncoder(), corpus_sha256="toy", batch_size=2,
                window_fn=lambda text: [text, text + " beta"])
    dense = DenseCorpus(corpus, tmp_path, ToyEncoder(), "dense", corpus_sha256="toy")
    results = dense.search("beta", 3)
    assert len(results) == 3
    assert {candidate.chunk_id for candidate in results} == {"a", "b", "c"}
    assert all("window" not in candidate.chunk_id for candidate in results)


def test_incomplete_index_corruption_is_not_blessed_on_resume(tmp_path):
    from research_agent.core.corpus.dense import build_index
    class Broken(ToyEncoder):
        count = 0
        def encode(self, texts):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("interrupted")
            return super().encode(texts)
    with pytest.raises(RuntimeError):
        build_index(toy_corpus(), tmp_path, Broken(), corpus_sha256="source", batch_size=2)
    vectors = np.load(tmp_path / "vectors.npy", mmap_mode="r+")
    vectors[0] = [0., 0.]
    vectors.flush()
    with pytest.raises(ValueError, match="checksum"):
        build_index(toy_corpus(), tmp_path, ToyEncoder(), corpus_sha256="source", batch_size=2)


def test_live_embedding_fingerprint_not_copied_blindly_from_index(tmp_path):
    from research_agent.core.corpus.dense import EmbeddingClient, file_hash
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"v1")
    client = EmbeddingClient("http://unused/v1", "toy", fingerprint={
        "root": str(model), "files": {"model.safetensors": file_hash(weights)}})
    weights.write_bytes(b"v2")
    with pytest.raises(Exception, match="fingerprint"):
        client.verify_server()


def test_window_indexing_through_real_embedding_http(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from research_agent.core.corpus.dense import EmbeddingClient, DenseCorpus, build_index
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send_json(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            self.send_json({"data": [{"id": "toy", "root": "toy-model", "max_model_len": 512}]})
        def do_POST(self):
            raw = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            items = [{"index": i, "embedding": [1., 0.] if text == "beta" or text == [2, 3] else [0., 1.]}
                     for i, text in enumerate(raw["input"])]
            self.send_json({"data": items[::-1], "usage": {"prompt_tokens": 20}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    encoder = EmbeddingClient(f"http://127.0.0.1:{server.server_port}/v1", "toy")
    try:
        corpus = toy_corpus()
        build_index(corpus, tmp_path, encoder, corpus_sha256="toy", batch_size=2,
                    window_fn=lambda text: [[2, 3]] if "semantic" in text else [[4, 5]])
        dense = DenseCorpus(corpus, tmp_path, encoder, "dense", corpus_sha256="toy")
        assert dense.search("beta", 1)[0].chunk_id == "b"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_local_tokenizer_is_used_for_queries_without_double_special_tokens(monkeypatch):
    from io import BytesIO
    from research_agent.core.corpus.dense import EmbeddingClient
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is True
            assert text == "The river."
            return [101, 100, 10835, 119, 102]
    requests = []
    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(b'{"data":[{"index":0,"embedding":[1,0]}]}')
    monkeypatch.setattr("urllib.request.urlopen", respond)
    client = EmbeddingClient("http://localhost/v1", "bge", tokenizer=Tokenizer())
    client.encode(["The river."])
    assert requests[-1]["input"] == [[101, 100, 10835, 119, 102]]
    assert requests[-1]["add_special_tokens"] is False
    client.encode([[101, 100, 10835, 119, 102]])
    assert requests[-1]["input"] == [[101, 100, 10835, 119, 102]]
    assert requests[-1]["add_special_tokens"] is False
    assert client.input_policy == "local_hf_token_ids_v1"


def test_index_rejects_different_query_tokenization_policy(tmp_path):
    from research_agent.core.corpus.dense import DenseCorpus, build_index
    first = ToyEncoder()
    first.input_policy = "local_hf_token_ids_v1"
    build_index(toy_corpus(), tmp_path, first, corpus_sha256="source")
    other = ToyEncoder()
    other.input_policy = "server_text_v1"
    with pytest.raises(ValueError, match="tokenization"):
        DenseCorpus(toy_corpus(), tmp_path, other, "dense", corpus_sha256="source")
