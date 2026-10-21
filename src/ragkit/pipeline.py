"""End-to-end pipeline: retrieve -> (refusal gate) -> generate, with latency logging."""
from __future__ import annotations
import hashlib, json, logging, os, pathlib, pickle, time
from dataclasses import dataclass, asdict
from .corpus import load_corpus
from .chunking import chunk_corpus
from .embedders import get_embedder
from .rerankers import get_reranker
from .retrieval import Index, Retriever
from .generate import get_generator, Generation, NOT_FOUND_MSG

log = logging.getLogger("ragkit")


@dataclass
class Config:
    raw_dir: str = "data/raw"
    chunk_tokens: int = 400
    overlap_frac: float = 0.15
    embedder: str = "BAAI/bge-small-en-v1.5"      # or "lsa" (offline stand-in)
    reranker: str | None = "BAAI/bge-reranker-base"   # None = off, or "lexical" (offline stand-in)
    mode: str = "hybrid"                          # dense | bm25 | hybrid
    k: int = 5
    candidates: int = 30
    generator: str = "extractive"                 # extractive | openai:<model>
    gate_threshold: float | None = None           # refuse if top gate_score < threshold; calibrate with eval/refusal_eval.py
    cache_dir: str = "artifacts"

    @classmethod
    def from_env(cls):
        e = os.environ.get
        c = cls()
        c.embedder = e("RAG_EMBEDDER", c.embedder)
        r = e("RAG_RERANKER", c.reranker or "none")
        c.reranker = None if r.lower() in ("none", "off", "") else r
        c.mode = e("RAG_MODE", c.mode)
        c.generator = e("RAG_GENERATOR", c.generator)
        c.chunk_tokens = int(e("RAG_CHUNK_TOKENS", c.chunk_tokens))
        c.cache_dir = e("RAG_CACHE_DIR", c.cache_dir)
        g = e("RAG_GATE")
        c.gate_threshold = float(g) if g else None
        return c


_INDEX_CACHE: dict = {}


def build_index(cfg: Config) -> Index:
    key = (cfg.raw_dir, cfg.chunk_tokens, cfg.overlap_frac, cfg.embedder)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    digest = hashlib.sha1(repr(key).encode()).hexdigest()[:12]   # stable across runs (hash() is not)
    path = pathlib.Path(cfg.cache_dir) / f"ix_{digest}.pkl"
    chunks = chunk_corpus(load_corpus(cfg.raw_dir), cfg.chunk_tokens, cfg.overlap_frac)
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if [c.id for c in saved["chunks"]] == [c.id for c in chunks]:
            emb_obj = saved["embedder"] or get_embedder(cfg.embedder)
            ix = Index(chunks, emb_obj, emb=saved["emb"])
            _INDEX_CACHE[key] = ix
            return ix
    emb_obj = get_embedder(cfg.embedder)
    ix = Index(chunks, emb_obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"chunks": chunks, "emb": ix.emb,
                                   "embedder": emb_obj if cfg.embedder.startswith("lsa") else None}))
    _INDEX_CACHE[key] = ix
    return ix


def build_retriever(cfg: Config, index: Index | None = None) -> Retriever:
    ix = index or build_index(cfg)
    rr = get_reranker(cfg.reranker, ix.bm25) if cfg.reranker else None
    return Retriever(ix, cfg.mode, rr, cfg.k, cfg.candidates)


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator, gate_threshold=None):
        self.retriever, self.generator, self.gate = retriever, generator, gate_threshold

    def answer(self, question: str, k: int | None = None) -> dict:
        t0 = time.perf_counter()
        ret = self.retriever.retrieve(question, k)
        top = ret.hits[0].gate_score if ret.hits else 0.0
        if self.gate is not None and top < self.gate:
            gen = Generation(NOT_FOUND_MSG, refused=True, refusal_reason="retrieval_gate")
        else:
            gen = self.generator.generate(question, ret.hits)
        total = (time.perf_counter() - t0) * 1e3
        out = {
            "question": question, "answer": gen.answer, "refused": gen.refused,
            "refusal_reason": gen.refusal_reason, "top_score": top,
            "citations": [{"chunk_id": h.chunk.id, "doc": h.chunk.doc_id, "section": h.chunk.heading_str,
                           "score": round(h.score, 4), "cited": h.chunk.id in gen.citations, "text": h.chunk.text}
                          for h in ret.hits],
            "invalid_citations": gen.invalid_citations,
            "latency_ms": {**{k: round(v, 1) for k, v in ret.timings_ms.items()},
                           "generate_ms": round(gen.generate_ms, 1), "total_ms": round(total, 1)},
        }
        log.info(json.dumps({"event": "answer", "refused": gen.refused, "top_score": round(top, 4),
                             "n_cited": len(gen.citations), **out["latency_ms"]}))
        return out


def build_pipeline(cfg: Config | None = None) -> RAGPipeline:
    cfg = cfg or Config.from_env()
    return RAGPipeline(build_retriever(cfg), get_generator(cfg.generator), cfg.gate_threshold)
