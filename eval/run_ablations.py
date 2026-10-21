"""Retrieval ablations: chunk size x {dense, bm25, hybrid} x reranker {off, on}."""
from __future__ import annotations
import argparse, json, pathlib, sys, time
sys.path.insert(0, "src")
from ragkit.pipeline import Config, build_index, build_retriever
from eval.metrics import load_questions, first_relevant_rank, retrieval_summary


def run(embedder, reranker, sizes, modes, k=10, out_tag="run"):
    qs = [q for q in load_questions() if q["type"] == "answerable"]
    rows = []
    for size in sizes:
        cfg0 = Config(embedder=embedder, reranker=None, chunk_tokens=size)
        ix = build_index(cfg0)                   # one index per chunk size, shared across configs
        for rerank_on in (False, True):
            for mode in modes:
                cfg = Config(embedder=embedder, reranker=reranker if rerank_on else None, chunk_tokens=size, mode=mode)
                ret = build_retriever(cfg, ix)
                ranks, ctx_words, t0 = [], [], time.perf_counter()
                for q in qs:
                    hits = ret.retrieve(q["question"], k).hits
                    ranks.append(first_relevant_rank(hits, q))
                    ctx_words.append(sum(h.chunk.n_words for h in hits[:5]))
                ms = (time.perf_counter() - t0) / len(qs) * 1e3
                s = retrieval_summary(ranks)
                rows.append({"chunk_tokens": size, "n_chunks": len(ix.chunks), "mode": mode, "rerank": rerank_on,
                             **s, "ctx_words@5": sum(ctx_words) / len(ctx_words), "ms_per_query": ms})
                print(f"{size:>4} {mode:<6} rerank={rerank_on!s:<5} R@5={s['recall@5']:.2f} MRR={s['mrr@10']:.2f}", flush=True)
    path = pathlib.Path("results"); path.mkdir(exist_ok=True)
    (path / f"ablation_{out_tag}.json").write_text(json.dumps({"embedder": embedder, "reranker": reranker, "rows": rows}, indent=1))
    (path / f"ablation_{out_tag}.md").write_text(to_markdown(rows, embedder, reranker))
    return rows


def to_markdown(rows, embedder, reranker):
    h = f"Embedder: `{embedder}`; reranker: `{reranker}`; n={rows[0]['n']} answerable questions; Recall@k = a gold chunk is in the top k.\n\n"
    h += "| Chunk (tok) | Retrieval | Rerank | R@1 | R@3 | R@5 [95% CI] | R@10 | MRR@10 | Ctx words@5 | ms/q |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        lo, hi = r["recall@5_ci95"]
        h += (f"| {r['chunk_tokens']} | {r['mode']} | {'yes' if r['rerank'] else 'no'} | {r['recall@1']:.2f} | {r['recall@3']:.2f} | "
              f"{r['recall@5']:.2f} [{lo:.2f}-{hi:.2f}] | {r['recall@10']:.2f} | {r['mrr@10']:.3f} | {r['ctx_words@5']:.0f} | {r['ms_per_query']:.0f} |\n")
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="LSA embedder + lexical reranker (no model downloads)")
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--sizes", type=int, nargs="+", default=[150, 300, 400, 500])
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    emb, rr = ("lsa", "lexical") if a.offline else (a.embedder, a.reranker)
    run(emb, rr, a.sizes, ["dense", "bm25", "hybrid"], out_tag=a.tag or ("offline" if a.offline else "real"))
