"""Metrics. Relevance is defined by gold *evidence phrases* inside a gold document, not by chunk id,
so labels stay valid when chunking changes (needed for the chunk-size ablation)."""
from __future__ import annotations
import json, re
import numpy as np

norm = lambda s: re.sub(r"\s+", " ", s).lower()


def load_questions(path="eval/questions.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def is_relevant(chunk, q) -> bool:
    if q["type"] != "answerable" or chunk.doc_id != q["doc"]:
        return False
    t = norm(chunk.text)
    return any(norm(e) in t for e in q["evidence"])


def first_relevant_rank(hits, q):
    for r, h in enumerate(hits, 1):
        if is_relevant(h.chunk, q):
            return r
    return None


def retrieval_summary(ranks, ks=(1, 3, 5, 10), mrr_cut=10, n_boot=2000, seed=0):
    """ranks: list of first-relevant rank (1-based) or None. Recall@k = share of questions with a gold chunk in top-k."""
    r = np.array([x if x else 10**9 for x in ranks])
    out = {f"recall@{k}": float((r <= k).mean()) for k in ks}
    rr = np.where(r <= mrr_cut, 1.0 / r, 0.0)
    out[f"mrr@{mrr_cut}"] = float(rr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), (n_boot, len(r)))
    out["recall@5_ci95"] = [float(x) for x in np.percentile((r[idx] <= 5).mean(1), [2.5, 97.5])]
    out["n"] = len(ranks)
    return out


def key_fact_recall(answer: str, facts) -> float:
    if not facts:
        return float("nan")
    a = answer.lower()
    return sum(f.lower() in a for f in facts) / len(facts)


def citation_scores(result: dict, q) -> dict:
    cited = [c for c in result["citations"] if c["cited"]]
    from types import SimpleNamespace as NS
    rel = [is_relevant(NS(doc_id=c["doc"], text=c["text"]), q) for c in cited]
    return {"cite_precision": (sum(rel) / len(rel)) if rel else 0.0, "cite_hit": float(any(rel)), "n_cited": len(cited)}


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")
