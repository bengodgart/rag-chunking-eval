"""A hand-rolled BM25 index in pure standard library.

No third-party dependency (no rank-bm25), no embedding model, no vector
database. Just tokenize, count term frequencies, compute IDF, and score with the
standard Okapi BM25 formula:

    score(q, d) = sum over terms t in q of
                  idf(t) * ( f(t,d) * (k1 + 1) )
                           / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

with the standard defaults k1 = 1.5 and b = 0.75, and the non-negative IDF
variant used by Lucene / rank-bm25:

    idf(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

where N is the number of chunks and n(t) the number of chunks containing t.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over a list of chunk strings."""

    def __init__(self, chunks: List[str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.tokens = [tokenize(c) for c in chunks]
        self.doc_len = [len(t) for t in self.tokens]
        n = len(self.tokens)
        self.n = n
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0

        # term frequency per chunk, and document frequency per term
        self.tf: List[dict] = []
        df: dict = {}
        for toks in self.tokens:
            counts: dict = {}
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            for term in counts:
                df[term] = df.get(term, 0) + 1
        self.df = df
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: List[str], index: int) -> float:
        if self.avgdl == 0:
            return 0.0
        tf = self.tf[index]
        dl = self.doc_len[index]
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        total = 0.0
        for term in query_tokens:
            f = tf.get(term)
            if not f:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * (f * (self.k1 + 1)) / (f + norm)
        return total

    def rank(self, query: str) -> List[Tuple[int, float]]:
        """Return (chunk_index, score) for every chunk, best score first.

        Ties keep the earlier chunk first (stable), so the ranking is fully
        deterministic across runs.
        """
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self.n)]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        return self.rank(query)[:k]
