import pathlib
import pytest
from fastapi.testclient import TestClient
from ragkit.pipeline import Config, build_pipeline
import ragkit.api as api

pytestmark = pytest.mark.skipif(not pathlib.Path("data/raw/clustering.rst").exists(),
                                reason="run scripts/fetch_corpus.py first")


def make_cfg(tmp_path, **kw):
    return Config(embedder="lsa", reranker="lexical", generator="extractive",
                  cache_dir=str(tmp_path / "cache"), **kw)


def test_pipeline_answers_with_citations(tmp_path):
    r = build_pipeline(make_cfg(tmp_path)).answer("Which two parameters define density in DBSCAN?")
    assert not r["refused"] and any(c["cited"] for c in r["citations"])
    assert all(k in r["latency_ms"] for k in ("retrieve_ms", "rerank_ms", "generate_ms", "total_ms"))


def test_gate_forces_refusal(tmp_path):
    cfg = make_cfg(tmp_path, gate_threshold=2.0)      # impossible threshold -> always refuse
    r = build_pipeline(cfg).answer("Which two parameters define density in DBSCAN?")
    assert r["refused"] and r["refusal_reason"] == "retrieval_gate"


def test_index_cache_is_reused_across_processes(tmp_path):
    from ragkit import pipeline
    cfg = make_cfg(tmp_path)
    pipeline._INDEX_CACHE.clear(); pipeline.build_index(cfg)
    files = sorted((tmp_path / "cache").glob("ix_*.pkl"))
    pipeline._INDEX_CACHE.clear(); pipeline.build_index(cfg)          # simulates a new run
    assert len(files) == 1 and sorted((tmp_path / "cache").glob("ix_*.pkl")) == files


def test_api_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_EMBEDDER", "lsa"); monkeypatch.setenv("RAG_RERANKER", "lexical")
    monkeypatch.setenv("RAG_GENERATOR", "extractive")
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "cache"))
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/ask", json={"question": "What does TruncatedSVD compute?"}).json()
        assert r["answer"] and r["citations"]
        assert c.post("/ask", json={"question": "hi"}).status_code == 422
        assert c.get("/stats").json()["requests"] == 1
