"""Answer generation with numbered citations and an honest not-found path.

The prompt is a fixed template. It must never contain evaluation questions or gold
answers (tests/test_no_leakage.py enforces this)."""
from __future__ import annotations
import json, os, re, time
from dataclasses import dataclass, field

NOT_FOUND_MSG = "I couldn't find this in the documentation I have access to."
NOT_FOUND_TOKEN = "NOT_FOUND"

SYSTEM_PROMPT = f"""You answer questions about the scikit-learn user guide using ONLY the numbered context passages provided.
Rules:
1. Use only information in the passages. Do not use outside knowledge, even if you know the answer.
2. Cite every factual sentence with the passage number(s) in square brackets, e.g. [1] or [2][3].
3. If the passages do not contain the answer, reply with exactly {NOT_FOUND_TOKEN} and nothing else. A partial or tangential match is NOT an answer.
4. Be concise: at most four sentences unless code is needed."""


def build_user_prompt(question: str, hits) -> str:
    ctx = "\n\n".join(f"[{i}] ({h.chunk.heading_str})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
    return f"Context passages:\n\n{ctx}\n\nQuestion: {question}"


@dataclass
class Generation:
    answer: str
    citations: list = field(default_factory=list)       # chunk ids actually cited
    refused: bool = False
    invalid_citations: list = field(default_factory=list)
    generate_ms: float = 0.0
    refusal_reason: str | None = None


def parse_citations(text: str, n_hits: int):
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    valid = sorted({n for n in nums if 1 <= n <= n_hits})
    invalid = sorted({n for n in nums if not 1 <= n <= n_hits})
    return valid, invalid


class ExtractiveGenerator:
    """Offline stand-in: returns the best-matching sentences verbatim with citations.
    No LLM, so it cannot hallucinate, but it also cannot synthesise."""
    name = "extractive-standin"

    def generate(self, question, hits) -> Generation:
        from .bm25 import tokenize
        t = time.perf_counter()
        qt = set(tokenize(question))
        scored = []
        for n, h in enumerate(hits[:3], 1):
            for sent in re.split(r"(?<=[.!?])\s+|\n\n", h.chunk.text):
                if len(sent.split()) < 5 or sent.lstrip().startswith((">>>", "...")):
                    continue
                st = set(tokenize(sent))
                scored.append((len(qt & st) / (len(qt) or 1), n, " ".join(sent.split())))
        scored.sort(key=lambda x: -x[0])
        picked = scored[:2]
        if not picked or picked[0][0] == 0:
            return Generation(NOT_FOUND_MSG, refused=True, refusal_reason="no_overlap",
                              generate_ms=(time.perf_counter() - t) * 1e3)
        text = " ".join(f"{s} [{n}]" for _, n, s in picked)
        valid, _ = parse_citations(text, len(hits))
        return Generation(text, [hits[n - 1].chunk.id for n in valid], False, [], (time.perf_counter() - t) * 1e3)


def _finish(text: str, hits, ms: float) -> Generation:
    if text.upper().startswith(NOT_FOUND_TOKEN):
        return Generation(NOT_FOUND_MSG, refused=True, generate_ms=ms, refusal_reason="model_not_found")
    valid, invalid = parse_citations(text, len(hits))
    return Generation(text, [hits[n - 1].chunk.id for n in valid], False, invalid, ms)


_LAST_CALL = [0.0]


def _cache_path(base, model, messages, max_tokens):
    d = os.getenv("RAG_LLM_CACHE", ".llm_cache")        # set RAG_LLM_CACHE="" to disable
    if not d:
        return None
    import hashlib, pathlib
    key = hashlib.sha256(json.dumps([base, model, messages, max_tokens], sort_keys=True).encode()).hexdigest()
    return pathlib.Path(d) / f"{key}.txt"


def openai_compat_chat(messages, model, max_tokens=500, retries=8) -> str:
    """POST to any OpenAI-compatible /chat/completions endpoint (Ollama, Groq, Gemini, ...).
    Env: RAG_OPENAI_BASE_URL (default: local Ollama), RAG_OPENAI_API_KEY (default: 'ollama')."""
    import requests
    retries = int(os.getenv("RAG_MAX_RETRIES", retries))   # interactive demos should fail fast (e.g. 2), batch evals can wait
    base = os.getenv("RAG_OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    key = os.getenv("RAG_OPENAI_API_KEY", "ollama")
    max_tokens = max(max_tokens, int(os.getenv("RAG_MAX_TOKENS", "0")))   # raise for "thinking" models whose hidden reasoning eats the budget
    cp = _cache_path(base, model, messages, max_tokens)
    if cp is not None and cp.exists():                    # temperature is 0, so a cached reply is a valid reply
        return cp.read_text(encoding="utf-8")
    gap = float(os.getenv("RAG_MIN_INTERVAL_S", "0"))     # throttle for free-tier requests-per-minute limits
    for attempt in range(retries):
        wait = _LAST_CALL[0] + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.monotonic()
        r = requests.post(f"{base}/chat/completions", timeout=300,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens})
        if r.status_code in (429, 503) and attempt < retries - 1:     # free tiers rate-limit: back off and retry
            time.sleep(min(65, 2 ** attempt * 4)); continue
        if r.status_code >= 400:      # surface the provider's explanation instead of a bare status code
            raise RuntimeError(f"{r.status_code} from {base} (model={model!r}): {r.text[:600]}")
        out = (r.json()["choices"][0]["message"]["content"] or "").strip()
        if cp is not None and out:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(out, encoding="utf-8")
        return out


class OpenAICompatGenerator:
    def __init__(self, model: str):
        self.model, self.name = model, f"openai-compat:{model}"

    def generate(self, question, hits) -> Generation:
        t = time.perf_counter()
        text = openai_compat_chat([{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": build_user_prompt(question, hits)}], self.model)
        return _finish(text, hits, (time.perf_counter() - t) * 1e3)


def get_generator(name: str):
    """extractive | openai:<model> (any OpenAI-compatible endpoint, e.g. openai:llama3.1:8b for Ollama)"""
    if name == "extractive":
        return ExtractiveGenerator()
    if name.startswith("openai:"):
        return OpenAICompatGenerator(name[len("openai:"):])
    raise ValueError(f"unknown generator {name!r}; use 'extractive' or 'openai:<model>'")
