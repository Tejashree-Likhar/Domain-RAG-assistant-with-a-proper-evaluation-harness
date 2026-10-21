"""Calibrate the LLM judge against YOUR labels.
  1. python -m eval.calibrate_judge make-sheet --results results/e2e_gate.jsonl --n 30
  2. Open eval/judge_labels.jsonl and fill human_faithfulness / human_relevance (0,1,2) by reading the passages.
  3. python -m eval.calibrate_judge score --judge openai:<model>
Reports exact agreement and quadratic-weighted Cohen's kappa per dimension. Rule of thumb: if kappa < ~0.6,
fix the judge prompt or don't use its numbers."""
from __future__ import annotations
import argparse, json, random, sys
sys.path.insert(0, "src")
from sklearn.metrics import cohen_kappa_score

SHEET = "eval/judge_labels.jsonl"


def make_sheet(results, n, seed=0):
    rows = [json.loads(l) for l in open(results) if json.loads(l).get("refused") is False]
    random.Random(seed).shuffle(rows)
    with open(SHEET, "w") as f:
        for r in rows[:n]:
            f.write(json.dumps({"id": r["id"], "question": r["question"], "passages": r["passages"], "answer": r["answer"],
                                "human_faithfulness": None, "human_relevance": None}) + "\n")
    print(f"wrote {min(n, len(rows))} items to {SHEET}")


def agreement(human, judge):
    out = {}
    for dim in ("faithfulness", "relevance"):
        h = [x[dim] for x in human]; j = [x[dim] for x in judge]
        out[dim] = {"exact_agreement": sum(a == b for a, b in zip(h, j)) / len(h),
                    "weighted_kappa": float(cohen_kappa_score(h, j, weights="quadratic", labels=[0, 1, 2])), "n": len(h)}
    return out


def score(judge_name):
    from eval.judge import get_judge
    items = [json.loads(l) for l in open(SHEET)]
    items = [i for i in items if i["human_faithfulness"] is not None and i["human_relevance"] is not None]
    if not items:
        sys.exit("no human labels found in " + SHEET)
    judge = get_judge(judge_name)
    jud = [judge.judge(i["question"], i["passages"], i["answer"]) for i in items]
    hum = [{"faithfulness": i["human_faithfulness"], "relevance": i["human_relevance"]} for i in items]
    print(json.dumps(agreement(hum, jud), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-sheet"); m.add_argument("--results", required=True); m.add_argument("--n", type=int, default=30)
    s = sub.add_parser("score"); s.add_argument("--judge", required=True, help="openai:<model> or proxy")
    a = ap.parse_args()
    make_sheet(a.results, a.n) if a.cmd == "make-sheet" else score(a.judge)
