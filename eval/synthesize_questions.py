"""Generate SYNTHETIC candidate questions with an LLM for you to spot-check.
Output: eval/synthetic_candidates.jsonl (reviewed=false). Review each line, delete bad ones, then copy
the keepers into eval/questions.jsonl (source='synthetic'), assigning a split.

Known bias: LLM-written questions tend to reuse the passage's wording, which flatters BM25. The prompt asks for
paraphrase, but keep your handwritten questions as the primary signal and report the two sources separately."""
from __future__ import annotations
import argparse, json, os, random, re, sys
sys.path.insert(0, "src")
from ragkit.corpus import load_corpus
from ragkit.chunking import chunk_corpus

PROMPT = """Below is a passage from the scikit-learn user guide.
Write ONE question a practitioner might ask whose answer is stated in this passage.
Constraints: paraphrase; do not copy rare terms verbatim unless a user would naturally type them; the question must make sense
without seeing the passage (no 'in this passage'). Also return a short verbatim phrase (5-15 words) from the passage that contains the answer.
Reply as JSON only: {{"question": "...", "evidence": "...", "key_fact": "..."}}

Passage:
{text}"""


def main(n, seed, out):
    from ragkit.generate import openai_compat_chat
    model = os.environ["RAG_SYNTH_MODEL"]       # model name on your OpenAI-compatible endpoint
    chunks = [c for c in chunk_corpus(load_corpus(), 400) if c.n_words > 80]
    random.Random(seed).shuffle(chunks)
    norm = lambda s: re.sub(r"\s+", " ", s).lower()
    kept = 0
    with open(out, "w") as f:
        for c in chunks:
            if kept >= n:
                break
            text = openai_compat_chat([{"role": "user", "content": PROMPT.format(text=c.text)}], model, max_tokens=300)
            try:
                d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            except Exception:
                continue
            if norm(d["evidence"]) not in norm(c.text):      # evidence must be verbatim, else discard
                continue
            f.write(json.dumps({"id": f"s{kept + 1:02d}", "type": "answerable", "question": d["question"], "doc": c.doc_id,
                                "evidence": [d["evidence"]], "key_facts": [d["key_fact"]], "source": "synthetic",
                                "split": "dev" if kept % 2 == 0 else "test", "reviewed": False}) + "\n")
            kept += 1
    print(f"wrote {kept} candidates to {out}; spot-check them before use")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/synthetic_candidates.jsonl")
    a = ap.parse_args(); main(a.n, a.seed, a.out)
