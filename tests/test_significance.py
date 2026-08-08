
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import significance as S


def test_two_sided_flags_negative_effect():
    """A reliably NEGATIVE effect must be flagged significant (regression test
    for the old one-sided bug that only flagged positive effects)."""
    per_query = {
        "a": {"m": [0.0] * 30},   # a is consistently worse than b by 0.2
        "b": {"m": [0.2] * 30},
    }
    rows = S.compare(per_query, "m", [("a", "b")])
    r = rows[0]
    assert r["delta"] < 0
    assert r["ci_excludes_zero"] is True
    assert r["significant"] is True     # significant despite a being worse


def test_no_difference_not_significant():
    per_query = {"a": {"m": [0.5] * 20}, "b": {"m": [0.5] * 20}}
    rows = S.compare(per_query, "m", [("a", "b")])
    assert rows[0]["significant"] is False


def test_holm_is_monotone_and_bounded():
    per_query = {
        "a": {"m": [1.0, 0.9, 1.0, 0.8, 1.0, 0.95]},
        "b": {"m": [0.0, 0.1, 0.0, 0.2, 0.0, 0.05]},
        "c": {"m": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]},
    }
    rows = S.compare(per_query, "m", [("a", "b"), ("a", "c")])
    for r in rows:
        assert 0.0 <= r["p_holm"] <= 1.0
        assert r["p_holm"] >= r["wilcoxon_p"] - 1e-9   # adjusted >= raw
