import json
import os
import platform
import time
from typing import Dict, List

import numpy as np

from . import config
from .data_utils import load_corpus, load_queries
from .pipeline import RetrievalPipeline
from . import metrics as M
from . import significance as S


def _dataset_paths(dataset: str):
    if dataset not in config.DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Choose from {list(config.DATASETS)}")
    return config.DATASETS[dataset]


def _environment() -> Dict:
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
    try:
        import torch
        env["torch_threads"] = torch.get_num_threads()
        env["torch"] = torch.__version__
    except Exception:
        env["torch_threads"] = None
    return env


def evaluate(dataset: str = config.DEFAULT_DATASET,
             dense_backend: str = "auto") -> Dict:
    corpus_path, queries_path = _dataset_paths(dataset)
    docs = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    pipe = RetrievalPipeline(docs, dense_backend=dense_backend)
    labels = pipe.describe()

    max_k = max(config.EVAL_KS)
    summary: Dict[str, Dict] = {}
    per_query: Dict[str, Dict[str, List[float]]] = {}

    for strat in pipe.strategies:
        # Warm-up (untimed): absorbs first-call model/thread initialisation so
        # latency is not distorted by whichever strategy runs first.
        pipe.retrieve(strat, queries[0]["query"], k=max_k)

        agg = {f"ndcg@{k}": 0.0 for k in config.EVAL_KS}
        agg.update({f"recall@{k}": 0.0 for k in config.EVAL_KS})
        agg["mrr"] = 0.0
        latencies = []
        pq = {"ndcg@10": [], "mrr": [], "recall@10": []}

        for q in queries:
            relevant = set(q["relevant"])
            t0 = time.perf_counter()
            ranked = pipe.retrieve(strat, q["query"], k=max_k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            for k in config.EVAL_KS:
                agg[f"ndcg@{k}"] += M.ndcg_at_k(ranked, relevant, k)
                agg[f"recall@{k}"] += M.recall_at_k(ranked, relevant, k)
            rr = M.reciprocal_rank(ranked, relevant)
            agg["mrr"] += rr
            pq["ndcg@10"].append(M.ndcg_at_k(ranked, relevant, 10))
            pq["recall@10"].append(M.recall_at_k(ranked, relevant, 10))
            pq["mrr"].append(rr)

        n = len(queries)
        for key in agg:
            agg[key] /= n
        latencies.sort()
        agg["latency_ms_mean"] = sum(latencies) / len(latencies)
        agg["latency_ms_p50"] = latencies[len(latencies) // 2]
        agg["latency_ms_p95"] = latencies[min(len(latencies) - 1,
                                              int(0.95 * len(latencies)))]
        agg["label"] = labels[strat]
        summary[strat] = agg
        per_query[strat] = pq

    sig_pairs = [
        ("dense", "bm25"),
        ("hybrid_rrf", "bm25"),
        ("hybrid_weighted", "bm25"),
        ("hybrid_rrf_ce", "bm25"),
        ("hybrid_rrf_ce", "hybrid_rrf"),
        ("hybrid_rrf_ce", "bm25_ce"),
        ("hybrid_rrf_ce", "dense_ce"),
        ("hybrid_rrf_ce", "hybrid_rrf_ce_fast"),
    ]
    sig_pairs = [(a, b) for a, b in sig_pairs if a in per_query and b in per_query]
    significance = {
        "ndcg@10": S.compare(per_query, "ndcg@10", sig_pairs),
        "mrr": S.compare(per_query, "mrr", sig_pairs),
    }

    return {
        "dataset": dataset,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "dense_backend": pipe.dense.backend,
        "dense_model": getattr(config, "DENSE_MODEL", "n/a")
        if pipe.dense.backend == "sentence-transformers" else "TF-IDF+SVD",
        "ann_backend": pipe.dense.ann_backend,
        "reranker_backend": pipe.cross_encoder.backend,
        "environment": _environment(),
        "strategies": summary,
        "per_query": per_query,
        "significance": significance,
        "candidate_recall": _candidate_recall(pipe, queries),
        "ann_recall": _ann_vs_exact_recall(pipe, queries),
        "ablation": _run_ablations(pipe, queries),
        "rerank_depth": _rerank_depth_sweep(pipe, queries),
        "dev_test": _dev_test_protocol(pipe, queries),
    }


def _candidate_recall(pipe, queries) -> Dict:
    """Recall of each FIRST stage at the depths that matter for reranking.

    The reranker cannot rank a gold passage it never receives, so Recall@50 of
    the first stage is the true ceiling for `*_ce` strategies.
    """
    depth = max(config.RERANK_DEPTH_GRID)
    stages = {
        "bm25": lambda q: pipe.bm25.search(q, depth),
        "dense": lambda q: pipe.dense.search(q, depth),
        "hybrid_rrf": lambda q: pipe.hybrid_rrf(q, depth),
    }
    out = {}
    for name, fn in stages.items():
        r10 = r50 = 0.0
        for q in queries:
            rel = set(q["relevant"])
            ranked = [d for d, _ in fn(q["query"])]
            r10 += M.recall_at_k(ranked, rel, 10)
            r50 += M.recall_at_k(ranked, rel, 50)
        n = len(queries)
        out[name] = {"recall@10": r10 / n, "recall@50": r50 / n}
    return out


def _ann_vs_exact_recall(pipe, queries) -> Dict:
    """How much recall the FAISS-HNSW approximation loses vs exact search."""
    dr = pipe.dense
    if not hasattr(dr, "doc_emb"):
        return {}
    emb = dr.doc_emb
    overlap = 0.0
    for q in queries:
        qv = dr._embed_query(q["query"]).reshape(-1)
        exact = set(np.argsort(-(emb @ qv))[:10].tolist())
        ann_ids = [i for i, _ in [(dr.ids.index(d), s)
                                  for d, s in dr.search(q["query"], 10)]]
        overlap += len(exact & set(ann_ids)) / 10.0
    return {"ann_recall@10_vs_exact": overlap / len(queries),
            "backend": dr.ann_backend}


def _run_ablations(pipe, queries) -> Dict:
    def score(rank_fn):
        nd = rc = 0.0
        for q in queries:
            rel = set(q["relevant"])
            ranked = [d for d, _ in rank_fn(q["query"])]
            nd += M.ndcg_at_k(ranked, rel, 10)
            rc += M.recall_at_k(ranked, rel, 10)
        n = len(queries)
        return nd / n, rc / n

    alpha_rows = []
    for a in config.ALPHA_GRID:
        nd, rc = score(lambda qq, a=a: pipe.hybrid_weighted(qq, 10, alpha=a))
        alpha_rows.append({"alpha": a, "ndcg@10": nd, "recall@10": rc})
    rrf_rows = []
    for kk in config.RRF_K_GRID:
        nd, rc = score(lambda qq, kk=kk: pipe.hybrid_rrf(qq, 10, rrf_k=kk))
        rrf_rows.append({"rrf_k": kk, "ndcg@10": nd, "recall@10": rc})
    return {"alpha": alpha_rows, "rrf_k": rrf_rows}


def _rerank_depth_sweep(pipe, queries) -> Dict:
    """Does the first stage matter after reranking, as a function of how many
    candidates the reranker sees?  CE scores of the deepest pool are computed
    once per query and reused for every depth (a CE pair score is independent of
    the pool size), so this is cheap.
    """
    depths = config.RERANK_DEPTH_GRID
    max_d = max(depths)
    stages = {
        "bm25_ce": lambda q: [d for d, _ in pipe.bm25.search(q, max_d)],
        "dense_ce": lambda q: [d for d, _ in pipe.dense.search(q, max_d)],
        "hybrid_rrf_ce": lambda q: [d for d, _ in pipe.hybrid_rrf(q, max_d)],
    }
    result = {name: {d: 0.0 for d in depths} for name in stages}
    for q in queries:
        rel = set(q["relevant"])
        for name, fn in stages.items():
            cands = fn(q["query"])                       # first-stage order
            scored = dict(pipe.cross_encoder.rerank(q["query"], cands, len(cands)))
            for d in depths:
                pool = cands[:d]
                ranked = sorted(pool, key=lambda c: scored.get(c, -1e9),
                                reverse=True)
                result[name][d] += M.ndcg_at_k(ranked, rel, 10)
    n = len(queries)
    return {name: [{"depth": d, "ndcg@10": result[name][d] / n} for d in depths]
            for name in stages}


def _dev_test_protocol(pipe, queries) -> Dict:
    """Split queries into dev/test; select weighted-fusion alpha on dev, report
    on the disjoint test split, alongside the fixed default alpha. Demonstrates
    the headline numbers are not the product of tuning on the test set.
    """
    rng = np.random.default_rng(config.SEED)
    idx = np.arange(len(queries))
    rng.shuffle(idx)
    cut = int(len(queries) * config.DEV_FRACTION)
    dev = [queries[i] for i in idx[:cut]]
    test = [queries[i] for i in idx[cut:]]

    def ndcg(qs, alpha):
        s = 0.0
        for q in qs:
            ranked = [d for d, _ in pipe.hybrid_weighted(q["query"], 10, alpha=alpha)]
            s += M.ndcg_at_k(ranked, set(q["relevant"]), 10)
        return s / len(qs)

    dev_scores = {a: ndcg(dev, a) for a in config.ALPHA_GRID}
    best_alpha = max(dev_scores, key=dev_scores.get)
    return {
        "n_dev": len(dev), "n_test": len(test),
        "selected_alpha": best_alpha,
        "dev_ndcg_at_selected": dev_scores[best_alpha],
        "test_ndcg_at_selected": ndcg(test, best_alpha),
        "test_ndcg_at_default": ndcg(test, config.HYBRID_ALPHA),
        "default_alpha": config.HYBRID_ALPHA,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_table(result: Dict) -> str:
    ks = config.EVAL_KS
    cols = (["Strategy"] + [f"nDCG@{k}" for k in ks]
            + [f"Recall@{k}" for k in ks]
            + ["MRR", "Lat.mean", "Lat.p50", "Lat.p95"])
    rows = []
    for strat, m in result["strategies"].items():
        row = [m["label"]]
        row += [f"{m[f'ndcg@{k}']:.3f}" for k in ks]
        row += [f"{m[f'recall@{k}']:.3f}" for k in ks]
        row += [f"{m['mrr']:.3f}", f"{m['latency_ms_mean']:.1f}",
                f"{m['latency_ms_p50']:.1f}", f"{m['latency_ms_p95']:.1f}"]
        rows.append(row)
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    def line(v): return "  ".join(x.ljust(widths[i]) for i, x in enumerate(v))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(cols), sep] + [line(r) for r in rows])


def _fmt_significance(result: Dict) -> str:
    out = []
    for metric, rows in result["significance"].items():
        out.append(f"\n[{metric}] paired bootstrap 95% CI, Wilcoxon p, "
                   f"Holm-adjusted p (two-sided; sig = CI excludes 0 AND Holm p<0.05)")
        out.append(f"  {'comparison':38s} {'d':>7}  {'95% CI':>18}  "
                   f"{'p':>7}  {'p_holm':>7}  sig")
        for r in rows:
            comp = f"{r['better']} vs {r['baseline']}"
            ci = f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
            p = r["wilcoxon_p"]; ph = r["p_holm"]
            ps = "  n/a" if p != p else f"{p:.4f}"
            phs = "  n/a" if ph != ph else f"{ph:.4f}"
            out.append(f"  {comp:38s} {r['delta']:+7.3f}  {ci:>18}  "
                       f"{ps:>7}  {phs:>7}  {'yes' if r['significant'] else 'no'}")
    return "\n".join(out)


def _fmt_extras(result: Dict) -> str:
    out = []
    cr = result["candidate_recall"]
    out.append("\nFirst-stage candidate recall (ceiling for the reranker):")
    out.append(f"  {'stage':12s} {'Recall@10':>9} {'Recall@50':>9}")
    for name, d in cr.items():
        out.append(f"  {name:12s} {d['recall@10']:>9.3f} {d['recall@50']:>9.3f}")

    ar = result.get("ann_recall") or {}
    if ar:
        out.append(f"\nANN vs exact ({ar.get('backend')}): dense top-10 recall "
                   f"vs brute-force = {ar['ann_recall@10_vs_exact']:.3f} "
                   f"(1.000 = no approximation loss)")

    rd = result["rerank_depth"]
    depths = [r["depth"] for r in next(iter(rd.values()))]
    out.append("\nRerank-depth sweep (nDCG@10 vs #candidates the reranker sees):")
    out.append("  " + "depth".ljust(14) + "  ".join(f"{d:>7}" for d in depths))
    for name, rows in rd.items():
        out.append("  " + name.ljust(14)
                   + "  ".join(f"{r['ndcg@10']:>7.3f}" for r in rows))

    dt = result["dev_test"]
    out.append(f"\nDev/test protocol (alpha selected on {dt['n_dev']} dev queries, "
               f"reported on {dt['n_test']} disjoint test queries):")
    out.append(f"  selected alpha (on dev)         : {dt['selected_alpha']}  "
               f"(dev nDCG@10 {dt['dev_ndcg_at_selected']:.3f})")
    out.append(f"  TEST nDCG@10 @ selected alpha    : {dt['test_ndcg_at_selected']:.3f}")
    out.append(f"  TEST nDCG@10 @ default alpha={dt['default_alpha']}  : "
               f"{dt['test_ndcg_at_default']:.3f}")

    a = result["ablation"]
    out.append("\nAblation: weighted-fusion alpha (dense weight)")
    out.append(f"  {'alpha':>6} {'nDCG@10':>8} {'Recall@10':>9}")
    for r in a["alpha"]:
        out.append(f"  {r['alpha']:>6.2f} {r['ndcg@10']:>8.3f} {r['recall@10']:>9.3f}")
    out.append("Ablation: RRF constant k")
    out.append(f"  {'rrf_k':>6} {'nDCG@10':>8} {'Recall@10':>9}")
    for r in a["rrf_k"]:
        out.append(f"  {r['rrf_k']:>6d} {r['ndcg@10']:>8.3f} {r['recall@10']:>9.3f}")
    return "\n".join(out)


def save_report(result: Dict) -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    tag = result["dataset"]

    with open(os.path.join(config.RESULTS_DIR, f"metrics_{tag}.json"), "w") as f:
        json.dump({k: v for k, v in result.items() if k != "per_query"}, f, indent=2)

    ks = config.EVAL_KS
    header = (["strategy", "label"] + [f"ndcg@{k}" for k in ks]
              + [f"recall@{k}" for k in ks]
              + ["mrr", "latency_ms_mean", "latency_ms_p50", "latency_ms_p95"])
    lines = [",".join(header)]
    for strat, m in result["strategies"].items():
        vals = [strat, f"\"{m['label']}\""]
        vals += [f"{m[f'ndcg@{k}']:.4f}" for k in ks]
        vals += [f"{m[f'recall@{k}']:.4f}" for k in ks]
        vals += [f"{m['mrr']:.4f}", f"{m['latency_ms_mean']:.4f}",
                 f"{m['latency_ms_p50']:.4f}", f"{m['latency_ms_p95']:.4f}"]
        lines.append(",".join(vals))
    with open(os.path.join(config.RESULTS_DIR, f"metrics_{tag}.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    env = result["environment"]
    report = (
        "Hybrid Retrieval on 3GPP Documents - Results\n"
        "Author: Kopal Sharma\n\n"
        f"Benchmark: {tag}  |  Corpus: {result['n_docs']} chunks  |  "
        f"Queries: {result['n_queries']}\n"
        f"Dense: {result['dense_model']} | ANN: {result['ann_backend']} | "
        f"Reranker: {result['reranker_backend']}\n"
        f"Env: {env['platform']} | py{env['python']} | "
        f"cpu={env['cpu_count']} | OMP={env['omp_num_threads']} | "
        f"torch_threads={env['torch_threads']}\n"
        f"(latency measured single-process after a warm-up call; "
        f"mean/p50/p95 over {result['n_queries']} queries)\n\n"
        + _fmt_table(result) + "\n"
        + _fmt_significance(result) + "\n"
        + _fmt_extras(result) + "\n"
    )
    with open(os.path.join(config.RESULTS_DIR, f"report_{tag}.txt"), "w") as f:
        f.write(report)
    _plots(result, tag)
    return report


def _plots(result: Dict, tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    strategies = list(result["strategies"].keys())
    labels = [result["strategies"][s]["label"] for s in strategies]
    ndcg = [result["strategies"][s]["ndcg@10"] for s in strategies]
    recall = [result["strategies"][s]["recall@10"] for s in strategies]
    lat = [result["strategies"][s]["latency_ms_mean"] for s in strategies]

    x = np.arange(len(strategies)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(x - w / 2, ndcg, w, label="nDCG@10")
    ax.bar(x + w / 2, recall, w, label="Recall@10")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Retrieval quality - {tag} benchmark "
                 f"({result['n_docs']} chunks) - Kopal Sharma")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, f"quality_{tag}.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(lat, ndcg, s=70)
    for i in range(len(strategies)):
        ax.annotate(labels[i], (lat[i], ndcg[i]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Mean latency per query (ms, warm)"); ax.set_ylabel("nDCG@10")
    ax.set_title(f"Quality vs latency - {tag} - Kopal Sharma")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, f"quality_vs_latency_{tag}.png"), dpi=140)
    plt.close(fig)

    rd = result["rerank_depth"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, rows in rd.items():
        ax.plot([r["depth"] for r in rows], [r["ndcg@10"] for r in rows],
                "o-", label=name)
    ax.set_xlabel("Rerank candidate depth"); ax.set_ylabel("nDCG@10")
    ax.set_title(f"Does the first stage matter after reranking? - {tag}")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, f"rerank_depth_{tag}.png"), dpi=140)
    plt.close(fig)


def main(dataset: str = config.DEFAULT_DATASET, dense_backend: str = "auto") -> None:
    result = evaluate(dataset=dataset, dense_backend=dense_backend)
    print(save_report(result))
    print(f"Results written to: {config.RESULTS_DIR} (tag: {dataset})")


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_DATASET
    be = sys.argv[2] if len(sys.argv) > 2 else "auto"
    main(ds, be)
