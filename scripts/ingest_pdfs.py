import json
import os
import re
import sys

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "raw")
CORPUS_PATH = os.path.join(os.path.dirname(RAW_DIR), "corpus.jsonl")

CHUNK_WORDS = 180
OVERLAP_WORDS = 40


def read_txt(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  ! pypdf not installed; skipping PDF. Run: pip install pypdf")
        return ""
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk(text: str):
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) < 30:
            break
        yield " ".join(piece)


def main():
    if not os.path.isdir(RAW_DIR):
        os.makedirs(RAW_DIR, exist_ok=True)
    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".pdf", ".txt"))]
    if not files:
        print(f"No .pdf/.txt files in {RAW_DIR}. Nothing to ingest.")
        return

    added = 0
    with open(CORPUS_PATH, "a", encoding="utf-8") as out:
        for fname in sorted(files):
            path = os.path.join(RAW_DIR, fname)
            spec = os.path.splitext(fname)[0]
            print(f"Ingesting {fname} ...")
            text = read_pdf(path) if fname.lower().endswith(".pdf") else read_txt(path)
            for i, ch in enumerate(chunk(text)):
                doc = {
                    "id": f"{spec}#chunk{i:04d}",
                    "spec": spec,
                    "section": f"chunk{i:04d}",
                    "title": f"{spec} (chunk {i})",
                    "text": ch,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                added += 1
    print(f"Appended {added} chunks to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
