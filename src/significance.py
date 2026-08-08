from typing import Dict, List

import numpy as np

from . import config


def paired_bootstrap_ci(a: List[float], b: List[float],
                        n_boot: int = None, seed: int = None):
    """95% CI of mean(a) - mean(b) via paired bootstrap resampling of queries."""
    n_boot = n_boot or config.BOOTSTRAP_SAMPLES
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def wilcoxon_p(a: List[float], b: List[float]) -> float:
    try:
        from scipy.stats import wilcoxon
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if np.allclose(a, b):
            return 1.0
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def _holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (order preserved)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (float("inf") if pvals[i] != pvals[i]
                                            else pvals[i]))
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        p = pvals[i]
        val = (m - rank) * p if p == p else float("nan")
        running = max(running, val) if val == val else running
        adj[i] = min(1.0, running) if val == val else float("nan")
    return adj


def compare(per_query: Dict[str, Dict[str, List[float]]],
            metric: str, pairs, alpha: float = 0.05) -> List[Dict]:
    """`per_query[strategy][metric]` -> comparison dicts (with Holm correction)."""
    rows = []
    for hi, lo in pairs:
        a = per_query[hi][metric]
        b = per_query[lo][metric]
        mean_d, ci_lo, ci_hi = paired_bootstrap_ci(a, b)
        p = wilcoxon_p(a, b)
        rows.append({
            "metric": metric, "better": hi, "baseline": lo,
            "delta": mean_d, "ci95": [ci_lo, ci_hi], "wilcoxon_p": p,
            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        })
    # Holm correction across this family of comparisons.
    adj = _holm([r["wilcoxon_p"] for r in rows])
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        r["significant"] = bool(r["ci_excludes_zero"]
                                and (pa < alpha if pa == pa else False))
    return rows
