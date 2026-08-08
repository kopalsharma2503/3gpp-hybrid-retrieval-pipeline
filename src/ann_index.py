"""Approximate nearest-neighbour index for dense retrieval.

Uses FAISS when available (HNSW graph index for sub-linear search), otherwise a
NumPy brute-force cosine fallback. Vectors are L2-normalized so inner product
equals cosine similarity.
"""

from typing import List, Tuple

import numpy as np


class ANNIndex:
    def __init__(self, embeddings: np.ndarray, use_faiss: bool = True):
        self.emb = embeddings.astype("float32")
        self.n, self.dim = self.emb.shape
        self.backend = "numpy"
        self._faiss = None

        if use_faiss:
            try:
                import faiss
                # HNSW: graph-based ANN, sub-linear query time at scale.
                index = faiss.IndexHNSWFlat(self.dim, 32,
                                            faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 128
                index.add(self.emb)
                self._faiss = index
                self.backend = "faiss-hnsw"
            except Exception:
                self._faiss = None
                self.backend = "numpy"

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        q = query_vec.astype("float32").reshape(1, -1)
        k = min(k, self.n)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]
        sims = (self.emb @ q.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
