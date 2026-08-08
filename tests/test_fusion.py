import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import hybrid


def test_rrf_rewards_agreement():
    # doc 'x' is ranked #1 by both retrievers -> should win the fusion.
    dense = [("x", 0.9), ("y", 0.8), ("z", 0.1)]
    sparse = [("x", 5.0), ("z", 4.0), ("y", 1.0)]
    fused = hybrid.reciprocal_rank_fusion([dense, sparse], k=3)
    assert fused[0][0] == "x"
    assert {d for d, _ in fused} == {"x", "y", "z"}


def test_weighted_alpha_extremes():
    dense = [("d1", 1.0), ("d2", 0.0)]
    sparse = [("s1", 1.0), ("s2", 0.0)]
    # alpha=1 -> dense only wins; alpha=0 -> sparse only wins.
    top_dense = hybrid.weighted_fusion(dense, sparse, k=1, alpha=1.0)[0][0]
    top_sparse = hybrid.weighted_fusion(dense, sparse, k=1, alpha=0.0)[0][0]
    assert top_dense == "d1"
    assert top_sparse == "s1"


def test_weighted_min_max_handles_constant_scores():
    # all-equal scores must not divide by zero.
    dense = [("a", 0.5), ("b", 0.5)]
    sparse = [("a", 2.0), ("c", 2.0)]
    fused = hybrid.weighted_fusion(dense, sparse, k=3, alpha=0.5)
    assert len(fused) == 3
