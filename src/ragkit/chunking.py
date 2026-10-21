"""Heading-aware chunking with overlap.

Size is expressed in approximate *model tokens*. We count whitespace words and use
1 word ~= 1.33 tokens, which avoids loading a tokenizer just to chunk. Paragraphs are
packed greedily so chunks rarely cut mid-paragraph; oversized paragraphs (long code
blocks, tables) fall back to word windows. Overlap carries trailing paragraphs forward.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .corpus import Section

TOKENS_PER_WORD = 1.33


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    headings: list
    text: str
    n_words: int

    @property
    def heading_str(self) -> str:
        return " > ".join(self.headings)

    @property
    def embed_text(self) -> str:
        """Text that gets embedded / reranked: heading context + body."""
        return f"{self.heading_str}\n{self.text}"

    def to_dict(self):
        return asdict(self)


def _wc(s: str) -> int:
    return len(s.split())


def _window_words(par: str, size: int, overlap: int):
    w = par.split()
    step = max(1, size - overlap)
    for start in range(0, len(w), step):
        yield " ".join(w[start:start + size])
        if start + size >= len(w):
            break


def chunk_section(sec: Section, max_words: int, overlap_words: int) -> list[str]:
    pars = [p for p in sec.text.split("\n\n") if p.strip()]
    units = []
    for p in pars:
        if _wc(p) > max_words:
            units.extend(_window_words(p, max_words, overlap_words))
        else:
            units.append(p)
    chunks, cur, cur_w = [], [], 0
    for u in units:
        uw = _wc(u)
        if cur and cur_w + uw > max_words:
            chunks.append("\n\n".join(cur))
            carry, cw = [], 0
            for prev in reversed(cur):          # overlap: trailing paragraphs up to overlap_words
                if cw + _wc(prev) > overlap_words:
                    break
                carry.insert(0, prev); cw += _wc(prev)
            cur, cur_w = carry, cw
        cur.append(u); cur_w += uw
    if cur and (not chunks or "\n\n".join(cur) != chunks[-1]):
        # avoid emitting a chunk that is purely overlap carried from the previous one
        if not chunks or any(u not in chunks[-1] for u in cur):
            chunks.append("\n\n".join(cur))
    return chunks


def chunk_corpus(sections: list[Section], chunk_tokens: int = 400, overlap_frac: float = 0.15) -> list[Chunk]:
    max_words = max(30, int(chunk_tokens / TOKENS_PER_WORD))
    overlap = int(max_words * overlap_frac)
    out, counter = [], {}
    for sec in sections:
        for text in chunk_section(sec, max_words, overlap):
            n = counter.get(sec.doc_id, 0); counter[sec.doc_id] = n + 1
            out.append(Chunk(f"{sec.doc_id}#{n}", sec.doc_id, sec.doc_title, sec.headings, text, _wc(text)))
    return out
