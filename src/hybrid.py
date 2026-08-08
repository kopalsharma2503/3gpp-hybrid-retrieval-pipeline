from typing import List, Tuple, Dict

from . import config


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    k: int,
    rrf_k: int = config.RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse several ranked lists using RRF: score = sum 1/(rrf_k + rank)."""
    scores: Dict[str, float] = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def _min_max(results: List[Tuple[str, float]]) -> Dict[str, float]:
    if not results:
        return {}
    vals = [s for _, s in results]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        return {d: 1.0 for d, _ in results}
    return {d: (s - lo) / span for d, s in results}


def weighted_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int,
    alpha: float = config.HYBRID_ALPHA,
) -> List[Tuple[str, float]]:
    """Weighted sum of min-max normalized dense and sparse scores.

    Documents missing from one list contribute 0 from that side.
    """
    d = _min_max(dense_results)
    s = _min_max(sparse_results)
    keys = set(d) | set(s)
    fused = {
        doc_id: alpha * d.get(doc_id, 0.0) + (1 - alpha) * s.get(doc_id, 0.0)
        for doc_id in keys
    }
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
