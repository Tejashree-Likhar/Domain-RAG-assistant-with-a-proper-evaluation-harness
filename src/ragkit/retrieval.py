"""Dense / BM25 / hybrid (RRF) retrieval with optional cross-encoder reranking."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from .bm25 import BM25
from .chunking import Chunk

try:
    import faiss
except ImportError:      # numpy brute force is exact too; FAISS just scales better
    faiss = None


class Index:
    """Chunks + dense matrix (FAISS flat inner-product, exact) + BM25."""
    def __init__(self, chunks: list[Chunk], embedder, emb=None):
        self.chunks, self.embedder = chunks, embedder
        texts = [c.embed_text for c in chunks]
        if emb is None:                      # `emb` given when loading a saved index
            embedder.fit(texts)
            emb = embedder.encode_docs(texts)
        self.emb = emb
        self.bm25 = BM25(texts)
        self.faiss = None
        if faiss is not None:
            self.faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self.faiss.add(self.emb)

    def dense_search(self, qv, n):
        if self.faiss is not None:
            s, i = self.faiss.search(qv[None, :].astype("float32"), n)
            return i[0], s[0]
        sims = self.emb @ qv
        idx = np.argsort(-sims)[:n]
        return idx, sims[idx]


@dataclass
class Hit:
    chunk: Chunk
    score: float            # score used for the final ordering
    gate_score: float       # comparable-scale score used for the refusal gate
    dense: float = 0.0
    bm25: float = 0.0
    rerank: float | None = None


@dataclass
class Retrieval:
    hits: list
    timings_ms: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, index: Index, mode="hybrid", reranker=None, k=5, candidates=30, rrf_k=60):
        assert mode in ("dense", "bm25", "hybrid")
        self.ix, self.mode, self.reranker, self.k, self.cand, self.rrf_k = index, mode, reranker, k, candidates, rrf_k

    def retrieve(self, query: str, k: int | None = None) -> Retrieval:
        k = k or self.k
        t0 = time.perf_counter()
        ix = self.ix
        qv = ix.embedder.encode_query(query)
        cos_all = ix.emb @ qv
        bm_all = ix.bm25.scores(query)
        d_idx, _ = ix.dense_search(qv, self.cand)
        b_idx = np.argsort(-bm_all)[: self.cand]
        if self.mode == "dense":
            order = [(int(i), float(cos_all[i])) for i in d_idx]
        elif self.mode == "bm25":
            order = [(int(i), float(bm_all[i])) for i in b_idx]
        else:  # reciprocal rank fusion
            fused = {}
            for lst in (d_idx, b_idx):
                for r, i in enumerate(lst):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.rrf_k + r + 1)
            order = sorted(fused.items(), key=lambda x: -x[1])[: self.cand]
        t1 = time.perf_counter()
        cands = [i for i, _ in order]
        base = dict(order)
        if self.reranker is not None:
            rs = self.reranker.score(query, [ix.chunks[i] for i in cands])
            ranked = sorted(zip(cands, rs), key=lambda x: -x[1])[:k]
            hits = [Hit(ix.chunks[i], float(s), float(s), float(cos_all[i]), float(bm_all[i]), float(s)) for i, s in ranked]
        else:
            top = cands[:k]
            gate = (lambda i: float(bm_all[i] / (bm_all.max() + 1e-9))) if self.mode == "bm25" else (lambda i: float(cos_all[i]))
            hits = [Hit(ix.chunks[i], base[i], gate(i), float(cos_all[i]), float(bm_all[i])) for i in top]
        t2 = time.perf_counter()
        return Retrieval(hits, {"retrieve_ms": (t1 - t0) * 1e3, "rerank_ms": (t2 - t1) * 1e3})
