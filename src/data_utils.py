"""Corpus / query loading and shared text utilities.

"""

import json
import re
from typing import List, Dict


def load_corpus(path: str) -> List[Dict]:
    """Load the corpus JSONL into a list of documents preserving order."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_queries(path: str) -> List[Dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def doc_text(doc: Dict) -> str:
    """Concatenate title and body so the title terms are searchable too."""
    return f"{doc.get('title', '')}. {doc.get('text', '')}".strip()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer used by the BM25 retriever."""
    return _TOKEN_RE.findall(text.lower())
