

from typing import List, Dict, Tuple

import numpy as np

from . import config
from .data_utils import doc_text


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class DenseRetriever:
    """Embeds passages and queries, ranks by cosine similarity."""

    def __init__(self, docs: List[Dict], backend: str = "auto",
                 use_faiss: bool = None):
        self.docs = docs
        self.ids = [d["id"] for d in docs]
        texts = [doc_text(d) for d in docs]
        self.backend = self._resolve_backend(backend)
        if use_faiss is None:
            use_faiss = config.USE_FAISS

        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.DENSE_MODEL)
            self.doc_emb = self.model.encode(
                texts, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
                batch_size=64,
            ).astype("float32")
        else:  # lsa fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            self.vectorizer = TfidfVectorizer(
                stop_words="english", sublinear_tf=True, ngram_range=(1, 2),
            )
            tfidf = self.vectorizer.fit_transform(texts)
            n_comp = min(128, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
            self.svd = TruncatedSVD(n_components=n_comp, random_state=config.SEED)
            self.doc_emb = _l2_normalize(
                self.svd.fit_transform(tfidf)).astype("float32")

        from .ann_index import ANNIndex
        self.index = ANNIndex(self.doc_emb, use_faiss=use_faiss)
        self.ann_backend = self.index.backend

    @property
    def name(self) -> str:
        if self.backend == "sentence-transformers":
            short = config.DENSE_MODEL.split("/")[-1]
            return f"Dense[{short}]"
        return "Dense[LSA]"

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend in ("sentence-transformers", "lsa"):
            return backend
        # auto: prefer the neural backend if it imports.
        try:
            import sentence_transformers  # noqa: F401
            import torch  # noqa: F401
            return "sentence-transformers"
        except Exception:
            return "lsa"

    def _embed_query(self, query: str) -> np.ndarray:
        if self.backend == "sentence-transformers":
            # Retrieval models such as BGE/E5 expect a query-side instruction;
            # passages are embedded without it (see __init__).
            q = getattr(config, "DENSE_QUERY_INSTRUCTION", "") + query
            return self.model.encode(
                [q], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
        vec = self.svd.transform(self.vectorizer.transform([query]))
        return _l2_normalize(vec)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        q = self._embed_query(query).reshape(-1)     # (dim,)
        hits = self.index.search(q, k)               # ANN (FAISS) or brute force
        return [(self.ids[i], score) for i, score in hits]
