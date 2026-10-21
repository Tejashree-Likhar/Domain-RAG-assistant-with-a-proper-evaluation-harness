"""Okapi BM25, no external dependency. Identifiers like n_clusters also index n, clusters."""
from __future__ import annotations
import math, re
from collections import Counter, defaultdict
import numpy as np

_TOK = re.compile(r"[a-z0-9_]+")
STOP = set("a an the of to in is are be it its for on and or as by with that this can how what which do does i my from at not use used using".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOK.findall(text.lower()):
        if t in STOP:
            continue
        out.append(t)
        if "_" in t:
            out.extend(p for p in t.split("_") if p and p not in STOP)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 0.0
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.post[t].append((i, f))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.post[t]:
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s
