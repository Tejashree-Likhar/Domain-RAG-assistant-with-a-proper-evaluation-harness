"""Embedding backends. Real runs use a sentence-transformers model (BGE / MiniLM);
`lsa` is an offline stand-in (TF-IDF + truncated SVD) used for tests and sandboxes
without model-hub access. Do not report LSA numbers as if they were BGE numbers."""
from __future__ import annotations
import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer   # lazy: heavy import
        self.name, self.model, self.bs = model_name, SentenceTransformer(model_name), batch_size
        self.q_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() and "en" in model_name.lower() else ""

    def fit(self, texts):  # no-op, API parity with LSA
        return self

    def encode_docs(self, texts):
        return np.asarray(self.model.encode(texts, batch_size=self.bs, normalize_embeddings=True,
                                            show_progress_bar=False), dtype="float32")

    def encode_query(self, q):
        return self.encode_docs([self.q_prefix + q])[0]


class LSAEmbedder:
    def __init__(self, dim: int = 128):
        self.name, self.dim = f"lsa-{dim}", dim

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=2)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(min(self.dim, X.shape[1] - 1), random_state=0).fit(X)
        return self

    def _enc(self, texts):
        Z = self.svd.transform(self.vec.transform(texts))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        return Z.astype("float32")

    encode_docs = _enc

    def encode_query(self, q):
        return self._enc([q])[0]


def get_embedder(name: str):
    return LSAEmbedder() if name.startswith("lsa") else SentenceTransformerEmbedder(name)
