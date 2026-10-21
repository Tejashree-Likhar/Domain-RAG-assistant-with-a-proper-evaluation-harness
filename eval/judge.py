"""Answer-quality judges.

OpenAICompatJudge: scores faithfulness (are all claims supported by the retrieved context?) and relevance
(does the answer address the question?) on a 0-2 scale. It must be calibrated against your own labels
(eval/calibrate_judge.py) before its numbers are trusted. It never sees gold answers or evidence.

ProxyJudge: offline lexical faithfulness proxy (share of answer content words found in the retrieved
context). Crude; use only when no LLM is available.
"""
from __future__ import annotations
import json, os, re
from ragkit.bm25 import tokenize

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented system.

Question: {question}

Retrieved context passages:
{context}

System answer:
{answer}

Score two things on a 0-2 scale:
faithfulness: 2 = every claim is supported by the context passages; 1 = mostly supported but at least one claim is unsupported or embellished; 0 = key claims are unsupported or contradict the context.
relevance: 2 = directly and completely answers the question; 1 = partially answers or is tangential; 0 = does not answer the question.
Judge faithfulness against the context only, NOT against your own knowledge.
Reply with JSON only: {{"faithfulness": <0|1|2>, "relevance": <0|1|2>, "reason": "<one sentence>"}}"""


def parse_judge_json(text: str) -> dict:
    """Strict JSON first; if the model's JSON is malformed (e.g. a stray \\u escape in 'reason'),
    fall back to regex-extracting the two scores. Raises ValueError if scores are missing."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"faithfulness": int(d["faithfulness"]), "relevance": int(d["relevance"]), "reason": str(d.get("reason", ""))}
        except Exception:
            pass
    f = re.search(r"faithfulness\W{1,6}([012])", text)
    r = re.search(r"relevance\W{1,6}([012])", text)
    if not (f and r):
        raise ValueError(f"could not parse judge reply: {text[:200]!r}")
    return {"faithfulness": int(f.group(1)), "relevance": int(r.group(1)), "reason": "(parsed leniently)"}


def format_context(passages):
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(passages, 1))


class ProxyJudge:
    name = "lexical-proxy"

    def judge(self, question, passages, answer) -> dict:
        ctx = set(tokenize(" ".join(passages)))
        toks = [t for t in tokenize(re.sub(r"\[\d+\]", " ", answer)) if not t.isdigit()]
        sup = sum(t in ctx for t in toks) / (len(toks) or 1)
        return {"faithfulness": sup, "relevance": None, "reason": "lexical proxy"}


class OpenAICompatJudge:
    """Same prompt and scale, via any OpenAI-compatible endpoint. Small local models make noisy judges:
    check kappa with eval/calibrate_judge.py before trusting the scores."""
    def __init__(self, model):
        self.model, self.name = model, f"openai-compat:{model}"

    def judge(self, question, passages, answer) -> dict:
        from ragkit.generate import openai_compat_chat
        text = openai_compat_chat([{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, context=format_context(passages), answer=answer)}], self.model, max_tokens=200)
        return parse_judge_json(text)


def get_judge(name: str):
    """proxy | openai:<model>"""
    if name == "proxy":
        return ProxyJudge()
    if name.startswith("openai:"):
        return OpenAICompatJudge(name[len("openai:"):])
    raise ValueError(f"unknown judge {name!r}; use 'proxy' or 'openai:<model>'")
