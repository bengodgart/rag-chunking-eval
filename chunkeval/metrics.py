"""The two retrieval metrics, defined exactly.

hit-rate@k
    The fraction of questions for which at least one relevant chunk appears in
    the top-k retrieved chunks.
        hit_rate@k = (number of questions with a relevant chunk in top-k) / (number of questions)

MRR@k (mean reciprocal rank, cut off at k)
    For each question, find the rank r of the first relevant chunk in the ranked
    list (1-based). Its reciprocal rank is 1/r if r <= k, otherwise 0. MRR@k is
    the mean of those reciprocal ranks over all questions.
        MRR@k = (1/Q) * sum over questions of ( 1/r if a relevant chunk is at rank r <= k else 0 )

"Relevant" is decided by :func:`is_relevant` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


def _normalize(text: str) -> str:
    """Lowercase and collapse runs of whitespace, so a span match does not depend
    on how a splitter happened to rejoin whitespace."""
    return " ".join(text.lower().split())


def is_relevant(chunk_text: str, chunk_doc_id: str, gold_doc_id: str, answer_span: Optional[str]) -> bool:
    """A chunk is relevant to a question when it comes from the gold document and,
    if the question carries an answer span, actually contains that span."""
    if chunk_doc_id != gold_doc_id:
        return False
    if answer_span:
        return _normalize(answer_span) in _normalize(chunk_text)
    return True


@dataclass
class QuestionResult:
    question_id: str
    first_relevant_rank: Optional[int]  # 1-based rank of first relevant chunk, or None
    hit: bool                            # was a relevant chunk within top-k
    reciprocal_rank: float               # 1/rank if hit else 0.0


def evaluate_question(ranked_relevance: List[bool], k: int) -> QuestionResult:
    """``ranked_relevance[i]`` is True when the chunk at rank i+1 is relevant.

    The list is the full ranking; the k cutoff is applied here.
    """
    first_rank: Optional[int] = None
    for i, rel in enumerate(ranked_relevance):
        if rel:
            first_rank = i + 1
            break
    hit = first_rank is not None and first_rank <= k
    rr = (1.0 / first_rank) if hit else 0.0
    return QuestionResult(question_id="", first_relevant_rank=first_rank, hit=hit, reciprocal_rank=rr)


def aggregate(results: List[QuestionResult]):
    """Return (hit_rate, mrr) over a list of per-question results."""
    if not results:
        return 0.0, 0.0
    q = len(results)
    hit_rate = sum(1 for r in results if r.hit) / q
    mrr = sum(r.reciprocal_rank for r in results) / q
    return hit_rate, mrr
