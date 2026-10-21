"""Exercise the OpenAI-compatible backend against a fake local server (no network, no key)."""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace as NS
import pytest
from ragkit.generate import get_generator, NOT_FOUND_MSG
from eval.judge import get_judge

REPLY = {"text": "Use min_samples and eps [1]."}


class H(BaseHTTPRequestHandler):
    calls = 0
    def do_POST(self):
        H.calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if H.calls == 1:                       # first call rate-limited: client must retry
            self.send_response(429); self.end_headers(); return
        txt = REPLY["text"] if "grading" not in body["messages"][-1]["content"] else '{"faithfulness": 2, "relevance": 1, "reason": "ok"}'
        out = json.dumps({"choices": [{"message": {"content": txt}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass


@pytest.fixture
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), H); H.calls = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("RAG_OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("RAG_LLM_CACHE", "")           # cache off unless a test turns it on
    yield srv
    srv.shutdown()


HITS = [NS(chunk=NS(id="clustering#1", heading_str="Clustering > DBSCAN", text="min_samples and eps"))]


def test_generator_retries_parses_citations_and_refuses(server):
    g = get_generator("openai:llama3.1:8b")
    out = g.generate("q", HITS)
    assert H.calls == 2 and out.citations == ["clustering#1"] and not out.refused
    REPLY["text"] = "NOT_FOUND"
    assert g.generate("q", HITS).answer == NOT_FOUND_MSG
    REPLY["text"] = "Use min_samples and eps [1]."


def test_judge_parses_json(server):
    j = get_judge("openai:llama3.1:8b").judge("q", ["p"], "a")
    assert j["faithfulness"] in (0, 1, 2) and j["relevance"] in (0, 1, 2)


def test_cache_avoids_repeat_calls(server, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_LLM_CACHE", str(tmp_path / "c"))
    g = get_generator("openai:m")
    a = g.generate("q", HITS); calls = H.calls
    b = g.generate("q", HITS)
    assert H.calls == calls and a.answer == b.answer


def test_judge_parser_survives_malformed_json():
    from eval.judge import parse_judge_json
    bad = '{"faithfulness": 2, "relevance": 1, "reason": "see \\u00zz escape"}'
    assert parse_judge_json(bad)["faithfulness"] == 2 and parse_judge_json(bad)["relevance"] == 1
    with pytest.raises(ValueError):
        parse_judge_json("no scores here")
