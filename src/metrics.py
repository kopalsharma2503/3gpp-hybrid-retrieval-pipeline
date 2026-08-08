import math
from typing import List, Set


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / k


def reciprocal_rank(ranked: List[str], relevant: Set[str]) -> float:
    for i, d in enumerate(ranked):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """nDCG with binary gains."""
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
