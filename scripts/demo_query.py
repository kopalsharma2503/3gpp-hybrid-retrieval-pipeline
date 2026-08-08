import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_utils import load_corpus
from src.pipeline import RetrievalPipeline


def main():
    query = " ".join(sys.argv[1:]).strip() or \
        "how does 5G-AKA authentication work?"

    # Prefer the real 3GPP corpus if it has been built; else the controlled set.
    real_c, _ = config.DATASETS["real"]
    corpus_path = real_c if os.path.exists(real_c) else config.CORPUS_PATH
    docs = load_corpus(corpus_path)
    title_by_id = {d["id"]: d["title"] for d in docs}
    pipe = RetrievalPipeline(docs)
    labels = pipe.describe()

    print(f"\nQuery: {query}\n" + "=" * 70)
    for strat in ["bm25", "dense", "hybrid_rrf", "hybrid_rrf_ce"]:
        print(f"\n[{labels[strat]}]")
        ranked = pipe.retrieve(strat, query, k=5)
        for i, doc_id in enumerate(ranked, 1):
            print(f"  {i}. {doc_id:18s}  {title_by_id.get(doc_id, '')}")


if __name__ == "__main__":
    main()
