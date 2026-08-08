
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics as M


def test_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d"}
    assert M.recall_at_k(ranked, rel, 2) == 0.5      # only b in top-2
    assert M.recall_at_k(ranked, rel, 4) == 1.0
    assert M.recall_at_k(ranked, set(), 4) == 0.0


def test_precision_at_k():
    ranked = ["a", "b", "c"]
    rel = {"a", "c"}
    assert M.precision_at_k(ranked, rel, 3) == 2 / 3
    assert M.precision_at_k(ranked, rel, 0) == 0.0


def test_reciprocal_rank():
    assert M.reciprocal_rank(["x", "y", "z"], {"y"}) == 0.5
    assert M.reciprocal_rank(["x", "y"], {"q"}) == 0.0


def test_ndcg_perfect_and_partial():
    rel = {"a", "b"}
    # perfect ranking -> nDCG 1.0
    assert abs(M.ndcg_at_k(["a", "b", "c"], rel, 3) - 1.0) < 1e-9
    # one relevant at rank 2 only
    ranked = ["c", "a", "d"]
    dcg = 1.0 / math.log2(2 + 1)          # 'a' at index 1
    idcg = 1.0 + 1.0 / math.log2(3)       # ideal: two rels at ranks 1,2
    assert abs(M.ndcg_at_k(ranked, rel, 3) - dcg / idcg) < 1e-9


def test_ndcg_empty_relevant():
    assert M.ndcg_at_k(["a", "b"], set(), 2) == 0.0
