"""End-to-end evaluation: retrieval + answers + refusal. Headline numbers are on the TEST split;
the refusal threshold is calibrated on DEV only."""
from __future__ import annotations
import argparse, json, pathlib, sys, statistics
sys.path.insert(0, "src")
from ragkit.pipeline import Config, build_retriever, RAGPipeline
from ragkit.generate import get_generator
from eval.metrics import load_questions, first_relevant_rank, key_fact_recall, citation_scores, mean
from eval.refusal_eval import evaluate as refusal_evaluate
from eval.judge import get_judge


def run_questions(pipe, qs, judge, k):
    rows = []
    for q in qs:
        r = pipe.answer(q["question"], k)
        row = {"id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"],
               "answer": r["answer"], "refused": r["refused"], "refusal_reason": r["refusal_reason"],
               "top_score": r["top_score"], "latency_ms": r["latency_ms"]["total_ms"],
               "passages": [c["text"] for c in r["citations"]], "cited_ids": [c["chunk_id"] for c in r["citations"] if c["cited"]]}
        if q["type"] == "answerable":
            from types import SimpleNamespace as NS
            hits = [NS(chunk=NS(doc_id=c["doc"], text=c["text"])) for c in r["citations"]]
            row["retrieval_rank"] = first_relevant_rank(hits, q)
            if not r["refused"]:
                row["key_fact_recall"] = key_fact_recall(r["answer"], q["key_facts"])
                row.update(citation_scores(r, q))
        if not r["refused"] and judge is not None:
            try:
                j = judge.judge(q["question"], row["passages"], r["answer"])
                row["faithfulness"], row["relevance"] = j["faithfulness"], j["relevance"]
            except Exception as e:           # one bad judge reply must not kill a long run; it is counted below
                row["judge_error"] = str(e)[:200]
                print(f"  judge failed on {q['id']}: {row['judge_error']}", flush=True)
        rows.append(row)
    return rows


def summarise(rows, split):
    R = [r for r in rows if r["split"] == split]
    A = [r for r in R if r["type"] == "answerable"]
    U = [r for r in R if r["type"] == "unanswerable"]
    ans = [r for r in A if not r["refused"]]
    lat = sorted(r["latency_ms"] for r in R)
    return {
        "split": split, "n_answerable": len(A), "n_unanswerable": len(U),
        "answerable_answered": len(ans) / len(A) if A else float("nan"),
        "false_refusal_rate": 1 - len(ans) / len(A) if A else float("nan"),
        "key_fact_recall": mean([r["key_fact_recall"] for r in ans]),
        "cite_hit": mean([r["cite_hit"] for r in ans]),
        "cite_precision": mean([r["cite_precision"] for r in ans]),
        "faithfulness": mean([r["faithfulness"] for r in A if "faithfulness" in r]),
        "relevance": mean([r["relevance"] for r in A if r.get("relevance") is not None]),
        "judge_failures": sum("judge_error" in r for r in R),
        "unanswerable_refused": mean([float(r["refused"]) for r in U]),
        "answered_when_unanswerable": mean([float(not r["refused"]) for r in U]),
        "latency_p50_ms": lat[len(lat) // 2], "latency_p95_ms": lat[min(len(lat) - 1, int(.95 * len(lat)))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--generator", default=None)
    ap.add_argument("--judge", default=None, help="proxy | openai:<model>")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--chunk-tokens", type=int, default=400)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="e2e")
    a = ap.parse_args()
    cfg = Config(mode=a.mode, chunk_tokens=a.chunk_tokens, k=a.k)
    if a.offline:
        cfg.embedder, cfg.reranker = "lsa", "lexical"
    if a.embedder:
        cfg.embedder = a.embedder
    if a.reranker:
        cfg.reranker = a.reranker
    if a.no_rerank:
        cfg.reranker = None
    gen_name = a.generator or "extractive"
    judge = get_judge(a.judge or "proxy")
    qs = load_questions()
    ret = build_retriever(cfg)
    gate_info, _ = refusal_evaluate(ret, qs)
    gen = get_generator(gen_name)
    report = {"config": vars(cfg) | {"generator": gen_name, "judge": judge.name}, "gate": gate_info, "variants": {}}
    for name, gate in (("no_gate (model/extractor decides)", None), ("with_gate (threshold from dev)", gate_info["threshold"])):
        rows = run_questions(RAGPipeline(ret, gen, gate), qs, judge, a.k)
        report["variants"][name] = {"dev": summarise(rows, "dev"), "test": summarise(rows, "test")}
        out = pathlib.Path("results"); out.mkdir(exist_ok=True)
        with open(out / f"{a.tag}_{'gate' if gate is not None else 'nogate'}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path(f"results/{a.tag}.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: {s: {m: round(v, 3) if isinstance(v, float) else v for m, v in d.items()}
                           for s, d in vv.items()} for k, vv in report["variants"].items()}, indent=1))
    print("threshold", gate_info["threshold"], "AUROC", round(gate_info["auroc_all"], 3))


if __name__ == "__main__":
    main()
