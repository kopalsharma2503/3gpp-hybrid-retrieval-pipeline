"""BM25 (sparse lexical) retriever.

"""

from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi

from .data_utils import doc_text, tokenize


class BM25Retriever:
    """Classic Okapi BM25 over the tokenized corpus."""

    name = "BM25"

    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.ids = [d["id"] for d in docs]
        self.tokens = [tokenize(doc_text(d)) for d in docs]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs, highest score first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
