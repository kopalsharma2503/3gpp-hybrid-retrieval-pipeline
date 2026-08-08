import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src import config
from src.data_utils import load_corpus, load_queries


@pytest.fixture(scope="module")
def controlled():
    c, q = config.DATASETS["controlled"]
    if not (os.path.exists(c) and os.path.exists(q)):
        from data.build_corpus import build
        build()
    return load_corpus(c), load_queries(q)


def test_pipeline_runs_lsa(controlled):
    from src.pipeline import RetrievalPipeline
    docs, queries = controlled
    pipe = RetrievalPipeline(docs, dense_backend="lsa")
    q = queries[0]["query"]
    for strat in pipe.strategies:
        ranked = pipe.retrieve(strat, q, k=5)
        assert 1 <= len(ranked) <= 5
        assert len(set(ranked)) == len(ranked)   # no duplicate doc ids


def test_lsa_beats_random_on_controlled(controlled):
    """Sanity: retrieval should find gold well above chance on the easy set."""
    from src.pipeline import RetrievalPipeline
    from src import metrics as M
    docs, queries = controlled
    pipe = RetrievalPipeline(docs, dense_backend="lsa")
    hits = 0
    for qq in queries:
        ranked = pipe.retrieve("hybrid_rrf", qq["query"], k=10)
        hits += M.recall_at_k(ranked, set(qq["relevant"]), 10)
    assert hits / len(queries) > 0.6
