"""Refusal behaviour: can the retrieval score tell answerable from unanswerable questions?
Threshold is chosen on the DEV split and reported on the TEST split (never tuned on test)."""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "src")
from sklearn.metrics import roc_auc_score
from eval.metrics import load_questions


def calibrate_threshold(scores, answerable):
    """Pick threshold maximising balanced accuracy (refuse when score < t). Returns (t, bal_acc)."""
    scores, y = np.asarray(scores), np.asarray(answerable, bool)
    cands = np.unique(scores)
    mids = np.concatenate([[cands[0] - 1e-6], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-6]])
    best = (None, -1)
    for t in mids:
        refuse = scores < t
        tpr = refuse[~y].mean() if (~y).any() else 0      # unanswerable correctly refused
        tnr = (~refuse[y]).mean() if y.any() else 0       # answerable correctly answered
        if (tpr + tnr) / 2 > best[1] + 1e-12:
            best = (float(t), (tpr + tnr) / 2)
    return best


def gate_scores(retriever, questions):
    return [retriever.retrieve(q["question"]).hits[0].gate_score for q in questions]


def evaluate(retriever, questions):
    s = np.array(gate_scores(retriever, questions))
    ans = np.array([q["type"] == "answerable" for q in questions])
    dev = np.array([q["split"] == "dev" for q in questions])
    thr, dev_bal = calibrate_threshold(s[dev], ans[dev])
    res = {"threshold": thr, "dev_balanced_acc": dev_bal, "auroc_all": float(roc_auc_score(ans, s))}
    for name, m in (("dev", dev), ("test", ~dev)):
        refuse = s[m] < thr
        res[name] = {"n_answerable": int(ans[m].sum()), "n_unanswerable": int((~ans[m]).sum()),
                     "unanswerable_refused": float(refuse[~ans[m]].mean()),
                     "answerable_wrongly_refused": float(refuse[ans[m]].mean())}
    return res, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--reranker", default=None)
    a = ap.parse_args()
    from ragkit.pipeline import Config, build_retriever
    cfg = Config(mode=a.mode)
    if a.offline:
        cfg.embedder, cfg.reranker = "lsa", "lexical"
    if a.embedder:
        cfg.embedder = a.embedder
    if a.reranker:
        cfg.reranker = a.reranker
    if a.no_rerank:
        cfg.reranker = None
    res, _ = evaluate(build_retriever(cfg), load_questions())
    print(json.dumps(res, indent=2))
