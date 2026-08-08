import json
import os
import random
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
CORPUS_OUT = os.path.join(HERE, "real_corpus.jsonl")
QUERIES_OUT = os.path.join(HERE, "real_queries.jsonl")

SEED = 42
CHUNK_WORDS = 160
OVERLAP_WORDS = 30
MAX_CHUNKS_PER_CLAUSE = 3      # bound corpus size / gold size
N_EVAL_QUERIES = 120          # sampled, reproducible eval set
MIN_CLAUSE_CHARS = 250        # a clause needs real body text to be a query

# clause heading like "5.3.3.4  Reception of the RRCSetup by the UE"
HEAD_RE = re.compile(r'^(\d{1,2}(?:\.\d{1,2}){1,3})\s+([A-Z][^\n]{2,70})$')

# Generic / boilerplate titles that make poor semantic queries.
STOP_TITLES = {
    "general", "introduction", "initiation", "void", "reserved", "scope",
    "references", "definitions", "abbreviations", "symbols", "foreword",
    "general requirements", "overview", "actions", "reception", "transmission",
    "definitions and abbreviations", "general description", "background",
}

# Lines that are ETSI/3GPP page furniture -> drop.
FURNITURE_RE = re.compile(
    r'^(ETSI|3GPP TS .*version .*Release|ETSI TS .*V\d|Release \d+|\d{1,4})\s*$'
)


def spec_from_filename(fname: str) -> str:
    m = re.search(r'ts_(\d{5,6})', fname)
    if not m:
        return os.path.splitext(fname)[0]
    d = m.group(1)
    # 38331 -> 38.331 ; 123501 -> 23.501 (strip an ETSI leading 1 if 6 digits)
    if len(d) == 6 and d[0] == "1":
        d = d[1:]
    return f"{d[:2]}.{d[2:]}"


def read_pdf_text(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)


def parse_sections(text: str):
    """Yield (clause, title, body) tuples from raw spec text."""
    clause = title = None
    body = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or FURNITURE_RE.match(line):
            continue
        m = HEAD_RE.match(line)
        if m and not line.endswith((".", ";", ":", ",")) \
                and len(m.group(2).split()) <= 11:
            if clause is not None:
                yield clause, title, " ".join(body)
            clause, title = m.group(1), m.group(2).strip()
            body = []
        else:
            if clause is not None:
                body.append(line)
    if clause is not None:
        yield clause, title, " ".join(body)


def chunk_words(text: str):
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    step = CHUNK_WORDS - OVERLAP_WORDS
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) < 25:
            break
        out.append(" ".join(piece))
        if len(out) >= MAX_CHUNKS_PER_CLAUSE:
            break
    return out


def build():
    pdfs = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {RAW_DIR}. Run scripts/download_3gpp.py first.")

    corpus = []
    # clause key includes spec so identical clause numbers across specs don't clash
    clause_chunks = defaultdict(list)   # (spec, clause) -> [chunk_id]
    clause_title = {}                   # (spec, clause) -> title
    clause_bodylen = defaultdict(int)   # (spec, clause) -> chars

    for fname in pdfs:
        spec = spec_from_filename(fname)
        print(f"Parsing TS {spec} ({fname}) ...")
        text = read_pdf_text(os.path.join(RAW_DIR, fname))
        n_sec = 0
        for clause, title, body in parse_sections(text):
            if len(body) < 60:
                continue
            chunks = chunk_words(body)
            if not chunks:
                continue
            n_sec += 1
            key = (spec, clause)
            clause_title[key] = title
            clause_bodylen[key] += len(body)
            for j, ch in enumerate(chunks):
                cid = f"{spec}#{clause}#{j}"
                corpus.append({
                    "id": cid, "spec": spec, "clause": clause,
                    "title": title, "text": ch,
                })
                clause_chunks[key].append(cid)
        print(f"  sections with content: {n_sec}")

    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(corpus)} chunks -> {CORPUS_OUT}")

    # Build candidate queries: substantive leaf clauses with real body text.
    candidates = []
    for key, cids in clause_chunks.items():
        spec, clause = key
        title = clause_title[key]
        if title.lower() in STOP_TITLES:
            continue
        if len(title) < 5 or len(title.split()) < 2:
            continue
        if clause_bodylen[key] < MIN_CLAUSE_CHARS:
            continue
        candidates.append((key, title, cids))

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    # stratify: round-robin across specs so the eval spans all specs
    by_spec = defaultdict(list)
    for c in candidates:
        by_spec[c[0][0]].append(c)
    picked, i = [], 0
    while len(picked) < min(N_EVAL_QUERIES, len(candidates)):
        progressed = False
        for spec in sorted(by_spec):
            if i < len(by_spec[spec]):
                picked.append(by_spec[spec][i])
                progressed = True
                if len(picked) >= N_EVAL_QUERIES:
                    break
        if not progressed:
            break
        i += 1

    with open(QUERIES_OUT, "w", encoding="utf-8") as f:
        for n, (key, title, cids) in enumerate(picked):
            spec, clause = key
            f.write(json.dumps({
                "qid": f"r{n:03d}", "query": title,
                "relevant": cids, "spec": spec, "clause": clause,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(picked)} auto queries -> {QUERIES_OUT}")

    specs = sorted({d["spec"] for d in corpus})
    print(f"Specs: {', '.join(specs)}")
    gsizes = [len(c[2]) for c in picked]
    if gsizes:
        print(f"Gold per query: min {min(gsizes)} / "
              f"mean {sum(gsizes)/len(gsizes):.1f} / max {max(gsizes)}")


if __name__ == "__main__":
    build()
