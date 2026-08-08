import os
import sys

from src import config
from src.evaluate import main as run_eval


def ensure_controlled():
    c, q = config.DATASETS["controlled"]
    if not (os.path.exists(c) and os.path.exists(q)):
        print("Building controlled corpus ...")
        from data.build_corpus import build
        build()


def ensure_real():
    c, _ = config.DATASETS["real"]
    if not os.path.exists(c):
        print("Real corpus not found. Building it from data/raw/*.pdf ...")
        raw = os.path.join(config.DATA_DIR, "raw")
        pdfs = [f for f in os.listdir(raw) if f.lower().endswith(".pdf")] \
            if os.path.isdir(raw) else []
        if not pdfs:
            print("No spec PDFs found. Downloading real 3GPP specs from ETSI ...")
            from scripts.download_3gpp import main as dl, DEFAULT_SPECS
            dl(DEFAULT_SPECS)
        from data.build_real_corpus import build
        build()
    # Natural-language query set (over the same corpus).
    nl = config.DATASETS["real_nl"][1]
    if not os.path.exists(nl):
        from data.build_nl_queries import build as build_nl
        build_nl()


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_DATASET
    backend = sys.argv[2] if len(sys.argv) > 2 else "auto"
    if dataset in ("real", "real_nl"):
        ensure_real()
    else:
        ensure_controlled()
    run_eval(dataset, backend)
