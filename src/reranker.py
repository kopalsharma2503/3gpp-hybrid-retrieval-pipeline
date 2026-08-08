import math
from typing import List, Tuple, Dict

from . import config
from .data_utils import doc_text, tokenize


class LexicalOverlapReranker:
    """IDF-weighted query/passage term-overlap reranker (no heavy deps)."""

    name = "LexicalRerank"

    def __init__(self, docs: List[Dict]):
        self.text_by_id = {d["id"]: doc_text(d) for d in docs}
        # Precompute IDF over the corpus so rare terms weigh more.
        n = len(docs)
        df: Dict[str, int] = {}
        for d in docs:
            for t in set(tokenize(doc_text(d))):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def _score(self, query: str, doc_id: str) -> float:
        q_terms = set(tokenize(query))
        d_terms = set(tokenize(self.text_by_id.get(doc_id, "")))
        return sum(self.idf.get(t, 0.0) for t in (q_terms & d_terms))

    def rerank(self, query: str, candidates: List[str], k: int) -> List[Tuple[str, float]]:
        scored = [(doc_id, self._score(query, doc_id)) for doc_id in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class CrossEncoderReranker:
    """Neural cross-encoder reranker with a lexical fallback.

    `model_name` selects which cross-encoder to load, so several reranking
    models of different capacity can be compared. `label` is a short display
    name used in the report.
    """

    def __init__(self, docs: List[Dict], model_name: str = None,
                 label: str = "CrossEncoder"):
        self.text_by_id = {d["id"]: doc_text(d) for d in docs}
        self.model_name = model_name or config.CROSS_ENCODER_MODEL
        self._label = label
        self.backend = "cross-encoder"
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.backend = "lexical"
            self._fallback = LexicalOverlapReranker(docs)

    @property
    def name(self) -> str:
        return self._label if self.backend == "cross-encoder" \
            else f"{self._label}(fallback=Lexical)"

    def rerank(self, query: str, candidates: List[str], k: int) -> List[Tuple[str, float]]:
        if self.backend == "lexical":
            return self._fallback.rerank(query, candidates, k)
        pairs = [(query, self.text_by_id.get(c, "")) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
