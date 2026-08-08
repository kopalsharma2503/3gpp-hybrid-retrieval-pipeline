"""Central configuration for the hybrid retrieval pipeline.
"""

import os


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----- Paths -----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

DATASETS = {
    "controlled": (os.path.join(DATA_DIR, "corpus.jsonl"),
                   os.path.join(DATA_DIR, "queries.jsonl")),

    "real": (os.path.join(DATA_DIR, "real_corpus.jsonl"),
             os.path.join(DATA_DIR, "real_queries.jsonl")),
    "real_nl": (os.path.join(DATA_DIR, "real_corpus.jsonl"),
                os.path.join(DATA_DIR, "nl_queries.jsonl")),
}
DEFAULT_DATASET = "real_nl"

# Back-compat aliases (used by the original controlled-set scripts).
CORPUS_PATH = DATASETS["controlled"][0]
QUERIES_PATH = DATASETS["controlled"][1]


DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # balanced
CROSS_ENCODER_MODEL_FAST = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # tiny/fast

# ----- Retrieval parameters -----
CANDIDATE_K = 50     # candidates each first-stage retriever returns
RERANK_K = 50        # candidates passed into the reranker
TOP_K = 10           # final cut-off used for reporting
RRF_K = 60           # Reciprocal Rank Fusion constant
HYBRID_ALPHA = 0.5   # weight on dense score in weighted fusion (0..1)

# ANN index: use FAISS for dense search when available (falls back to brute force).
USE_FAISS = True

# Metric cut-offs to report.
EVAL_KS = [1, 3, 5, 10]

# ----- Ablation grids -----
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]   # weighted-fusion dense weight
RRF_K_GRID = [10, 30, 60, 100]             # RRF constant
RERANK_DEPTH_GRID = [5, 10, 20, 50, 100]   # candidate depth fed to the reranker

# Headline results use these FIXED defaults (HYBRID_ALPHA, RRF_K), which are NOT
# selected from the ablation sweeps. A separate dev/test protocol (below) selects
# alpha on a dev split and reports on a disjoint test split to rule out tuning
# on the test set.
DEV_FRACTION = 0.5     # fraction of queries used as the dev (tuning) split

# Statistical significance.
BOOTSTRAP_SAMPLES = 5000
SEED = 42
