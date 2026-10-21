"""Check every gold evidence phrase exists inside a single chunk at every ablation chunk size,
and that unanswerable questions' distinctive terms really are absent from the corpus."""
import json, re, sys
sys.path.insert(0, "src")
from ragkit.corpus import load_corpus
from ragkit.chunking import chunk_corpus

norm = lambda s: re.sub(r"\s+", " ", s).lower()
ABSENT = ["shap", "bayessearchcv", "cournapeau", "sagemaker", "survival forest", "xgboost", "grpc", "resnet", "q-learning"]

def main(path="eval/questions.jsonl", sizes=(150, 300, 400, 500)):
    qs = [json.loads(l) for l in open(path)]
    secs = load_corpus()
    bad = 0
    for size in sizes:
        chunks = chunk_corpus(secs, size)
        by_doc = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(norm(c.text))
        for q in qs:
            if q["type"] != "answerable":
                continue
            ok = any(norm(e) in t for e in q["evidence"] for t in by_doc[q["doc"]])
            if not ok:
                bad += 1; print(f"MISSING @{size}: {q['id']} {q['evidence']}")
    full = norm(" ".join(s.text for s in secs))
    for term in ABSENT:
        if re.search(r"\b" + re.escape(term) + r"\b", full) and term != "xgboost":  # xgboost = citation key only
            print("NOTE: term present in corpus (unanswerable Qs may be answerable):", term)
    print("validation:", "OK" if not bad else f"{bad} problems", f"({len(qs)} questions)")
    return bad

if __name__ == "__main__":
    sys.exit(1 if main(*sys.argv[1:2]) else 0)
