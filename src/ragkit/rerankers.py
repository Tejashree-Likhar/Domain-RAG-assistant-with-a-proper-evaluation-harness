"""Rerankers. `CrossEncoderReranker` is the real thing (BGE reranker / MS MARCO MiniLM).
`LexicalReranker` is an offline stand-in: IDF-weighted query-term coverage plus bigram and
heading matches, scaled to 0..1. It exists so the harness runs without model downloads."""
from __future__ import annotations
import numpy as np
from .bm25 import tokenize


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.name, self.model = model_name, CrossEncoder(model_name)

    def score(self, query, chunks):
        s = self.model.predict([(query, c.embed_text) for c in chunks], show_progress_bar=False)
        return 1 / (1 + np.exp(-np.asarray(s, dtype=float)))      # sigmoid -> 0..1 so gating thresholds are comparable


class LexicalReranker:
    name = "lexical-standin"

    def __init__(self, idf: dict):
        self.idf = idf

    def score(self, query, chunks):
        q = tokenize(query)
        qset = set(q)
        qbi = set(zip(q, q[1:]))
        denom = sum(self.idf.get(t, 1.0) for t in qset) or 1.0
        out = []
        for c in chunks:
            toks = tokenize(c.embed_text)
            tset = set(toks)
            cov = sum(self.idf.get(t, 1.0) for t in qset if t in tset) / denom
            bi = len(qbi & set(zip(toks, toks[1:]))) / (len(qbi) or 1)
            head = len(qset & set(tokenize(c.heading_str))) / (len(qset) or 1)
            out.append(0.65 * cov + 0.25 * bi + 0.10 * head)
        return np.asarray(out)


def get_reranker(name: str, bm25=None):
    return LexicalReranker(bm25.idf) if name.startswith("lexical") else CrossEncoderReranker(name)
