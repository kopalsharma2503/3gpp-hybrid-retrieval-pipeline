import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_utils import load_corpus, load_queries
from src.pipeline import RetrievalPipeline


def rank_of_first_gold(ranked, gold):
    for i, d in enumerate(ranked):
        if d in gold:
            return i + 1
    return None


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_DATASET
    corpus_path, queries_path = config.DATASETS[tag]
    docs = load_corpus(corpus_path)
    title = {d["id"]: d.get("title", "") for d in docs}
    queries = load_queries(queries_path)
    pipe = RetrievalPipeline(docs)

    out = [f"Error analysis - {tag} benchmark - Kopal Sharma", "=" * 60]

    # 1) failures of the strongest first stage (hybrid weighted)
    out.append("\n## Hybrid-weighted FAILURES (gold not in top-10)")
    fails = 0
    for q in queries:
        gold = set(q["relevant"])
        ranked = pipe.retrieve("hybrid_weighted", q["query"], k=10)
        if rank_of_first_gold(ranked, gold) is None:
            fails += 1
            out.append(f"\n  Q: {q['query']}")
            out.append(f"     gold clause {q.get('clause','?')} "
                       f"({q.get('spec','?')}): {title.get(q['relevant'][0],'')[:60]}")
            out.append(f"     got: " + " | ".join(title.get(d, d)[:32] for d in ranked[:3]))
    out.append(f"\n  ({fails}/{len(queries)} hybrid-weighted failures@10)")

    # 2) BM25 vs Dense disagreements (one finds gold in top-3, the other misses@10)
    out.append("\n\n## BM25 vs DENSE disagreements")
    shown = 0
    for q in queries:
        gold = set(q["relevant"])
        b = pipe.retrieve("bm25", q["query"], k=10)
        d = pipe.retrieve("dense", q["query"], k=10)
        rb, rd = rank_of_first_gold(b, gold), rank_of_first_gold(d, gold)
        if rb and rd and ((rb <= 3 and rd is None) or (rd <= 3 and rb is None)
                          or (rb and rd and abs(rb - rd) >= 5)):
            winner = "BM25" if (rd is None or (rb and rb < rd)) else "Dense"
            out.append(f"\n  Q: {q['query']}")
            out.append(f"     BM25 gold rank={rb}  Dense gold rank={rd}  -> {winner} wins")
            shown += 1
        if shown >= 6:
            break

    # 3) reranker lifts (cross-encoder moves gold up vs hybrid first stage)
    out.append("\n\n## Cross-encoder RERANK lifts (gold rank before -> after)")
    shown = 0
    for q in queries:
        gold = set(q["relevant"])
        pre = pipe.retrieve("hybrid_rrf", q["query"], k=50)
        post = pipe.retrieve("hybrid_rrf_ce", q["query"], k=50)
        rpre, rpost = rank_of_first_gold(pre, gold), rank_of_first_gold(post, gold)
        if rpre and rpost and rpre - rpost >= 3:
            out.append(f"\n  Q: {q['query']}")
            out.append(f"     gold rank {rpre} -> {rpost} after cross-encoder")
            shown += 1
        if shown >= 6:
            break

    path = os.path.join(config.RESULTS_DIR, f"error_analysis_{tag}.txt")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
