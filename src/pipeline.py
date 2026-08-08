from typing import List, Dict, Tuple

from . import config
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .reranker import CrossEncoderReranker, LexicalOverlapReranker
from . import hybrid


class RetrievalPipeline:
    def __init__(self, docs: List[Dict], dense_backend: str = "auto"):
        self.docs = docs
        self.bm25 = BM25Retriever(docs)
        self.dense = DenseRetriever(docs, backend=dense_backend)
        self.cross_encoder = CrossEncoderReranker(
            docs, config.CROSS_ENCODER_MODEL, label="CE-MiniLM-L6")
        self.cross_encoder_fast = CrossEncoderReranker(
            docs, config.CROSS_ENCODER_MODEL_FAST, label="CE-TinyBERT-L2")
        self.lexical = LexicalOverlapReranker(docs)

        self.strategies = [
            "bm25",
            "dense",
            "hybrid_rrf",
            "hybrid_weighted",
            # --- controls: same reranker on a single first stage, to test
            #     whether hybrid candidates actually help the reranker ---
            "bm25_ce",
            "dense_ce",
            # --- full pipelines ---
            "hybrid_rrf_ce",
            "hybrid_rrf_ce_fast",
            "hybrid_rrf_lex",
        ]

    # -- first-stage helpers (with tunable params for ablations) ------------ #
    def _dense_bm25(self, query: str):
        d = self.dense.search(query, config.CANDIDATE_K)
        s = self.bm25.search(query, config.CANDIDATE_K)
        return d, s

    def hybrid_rrf(self, query: str, k: int, rrf_k: int = config.RRF_K):
        d, s = self._dense_bm25(query)
        return hybrid.reciprocal_rank_fusion([d, s], k, rrf_k=rrf_k)

    def hybrid_weighted(self, query: str, k: int, alpha: float = config.HYBRID_ALPHA):
        d, s = self._dense_bm25(query)
        return hybrid.weighted_fusion(d, s, k, alpha=alpha)

    # -- public: run one strategy for one query ----------------------------- #
    def retrieve(self, strategy: str, query: str, k: int = config.TOP_K) -> List[str]:
        if strategy == "bm25":
            ranked = self.bm25.search(query, k)
        elif strategy == "dense":
            ranked = self.dense.search(query, k)
        elif strategy == "hybrid_rrf":
            ranked = self.hybrid_rrf(query, k)
        elif strategy == "hybrid_weighted":
            ranked = self.hybrid_weighted(query, k)
        elif strategy == "bm25_ce":
            cand = [d for d, _ in self.bm25.search(query, config.RERANK_K)]
            ranked = self.cross_encoder.rerank(query, cand, k)
        elif strategy == "dense_ce":
            cand = [d for d, _ in self.dense.search(query, config.RERANK_K)]
            ranked = self.cross_encoder.rerank(query, cand, k)
        elif strategy == "hybrid_rrf_ce":
            cand = [d for d, _ in self.hybrid_rrf(query, config.RERANK_K)]
            ranked = self.cross_encoder.rerank(query, cand, k)
        elif strategy == "hybrid_rrf_ce_fast":
            cand = [d for d, _ in self.hybrid_rrf(query, config.RERANK_K)]
            ranked = self.cross_encoder_fast.rerank(query, cand, k)
        elif strategy == "hybrid_rrf_lex":
            cand = [d for d, _ in self.hybrid_rrf(query, config.RERANK_K)]
            ranked = self.lexical.rerank(query, cand, k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return [doc_id for doc_id, _ in ranked]

    def describe(self) -> Dict[str, str]:
        """Human-readable labels used in the report."""
        return {
            "bm25": "BM25 (sparse)",
            "dense": f"{self.dense.name} (semantic, {self.dense.ann_backend})",
            "hybrid_rrf": "Hybrid Dense+BM25 (RRF)",
            "hybrid_weighted": "Hybrid Dense+BM25 (weighted)",
            "bm25_ce": f"BM25 -> {self.cross_encoder.name}  [control]",
            "dense_ce": f"Dense -> {self.cross_encoder.name}  [control]",
            "hybrid_rrf_ce": f"Hybrid RRF -> {self.cross_encoder.name}",
            "hybrid_rrf_ce_fast": f"Hybrid RRF -> {self.cross_encoder_fast.name}",
            "hybrid_rrf_lex": f"Hybrid RRF -> {self.lexical.name}",
        }
