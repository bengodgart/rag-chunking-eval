"""Run one chunking config, and sweep the whole grid.

For a single config the pipeline is: split every document into chunks, build one
in-memory BM25 index over all chunks, retrieve top-k chunks per question, decide
relevance, and aggregate hit-rate@k and MRR@k. The sweep just does that for every
config in the grid and collects the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .bm25 import BM25Index
from .corpus import Doc, Question
from .metrics import QuestionResult, aggregate, evaluate_question, is_relevant
from .splitters import SPLITTERS


@dataclass(frozen=True)
class Config:
    splitter: str
    size: int
    overlap: int

    def label(self) -> str:
        return f"{self.splitter}/{self.size}/{self.overlap}"


@dataclass
class Chunk:
    doc_id: str
    text: str


@dataclass
class ConfigResult:
    config: Config
    num_chunks: int
    hit_rate: float
    mrr: float
    k: int
    per_question: List[QuestionResult] = field(default_factory=list)


# Small, explicit default grid: 3 splitters x 2 sizes x 2 overlaps = 12 configs.
DEFAULT_SIZES = [128, 256]
DEFAULT_OVERLAPS = [0, 64]
DEFAULT_SPLITTERS = ["fixed", "sentence", "recursive"]


def default_grid() -> List[Config]:
    grid: List[Config] = []
    for splitter in DEFAULT_SPLITTERS:
        for size in DEFAULT_SIZES:
            for overlap in DEFAULT_OVERLAPS:
                grid.append(Config(splitter=splitter, size=size, overlap=overlap))
    return grid


def build_chunks(docs: List[Doc], config: Config) -> List[Chunk]:
    splitter = SPLITTERS[config.splitter]
    chunks: List[Chunk] = []
    for doc in docs:
        for piece in splitter(doc.text, config.size, config.overlap):
            if piece.strip():
                chunks.append(Chunk(doc_id=doc.doc_id, text=piece))
    return chunks


def run_config(docs: List[Doc], questions: List[Question], config: Config, k: int) -> ConfigResult:
    chunks = build_chunks(docs, config)
    index = BM25Index([c.text for c in chunks])
    results: List[QuestionResult] = []
    for q in questions:
        ranked = index.rank(q.question)  # [(chunk_index, score), ...] best first
        ranked_relevance = [
            is_relevant(chunks[i].text, chunks[i].doc_id, q.doc_id, q.answer_span)
            for i, _ in ranked
        ]
        qr = evaluate_question(ranked_relevance, k)
        qr.question_id = q.id
        results.append(qr)
    hit_rate, mrr = aggregate(results)
    return ConfigResult(
        config=config,
        num_chunks=len(chunks),
        hit_rate=hit_rate,
        mrr=mrr,
        k=k,
        per_question=results,
    )


def run_sweep(docs: List[Doc], questions: List[Question], grid: List[Config], k: int) -> List[ConfigResult]:
    return [run_config(docs, questions, config, k) for config in grid]


def pick_winner(results: List[ConfigResult]) -> ConfigResult:
    """Highest hit-rate wins; MRR breaks a tie; fewer chunks breaks a further tie."""
    return max(results, key=lambda r: (r.hit_rate, r.mrr, -r.num_chunks))


@dataclass
class OverlapFinding:
    text: str
    helped_anywhere: bool
    per_family: List[str]


def overlap_finding(results: List[ConfigResult]) -> OverlapFinding:
    """Compare overlap 0 vs the largest overlap, holding splitter + size fixed,
    and state plainly whether overlap changed hit-rate anywhere."""
    by_key = {(r.config.splitter, r.config.size, r.config.overlap): r for r in results}
    overlaps = sorted({r.config.overlap for r in results})
    if len(overlaps) < 2:
        return OverlapFinding("Only one overlap value in the grid, nothing to compare.", False, [])
    lo, hi = overlaps[0], overlaps[-1]

    lines: List[str] = []
    helped = False
    families = sorted({(r.config.splitter, r.config.size) for r in results})
    for splitter, size in families:
        a = by_key.get((splitter, size, lo))
        b = by_key.get((splitter, size, hi))
        if a is None or b is None:
            continue
        delta = b.hit_rate - a.hit_rate
        if delta > 1e-9:
            helped = True
            verdict = f"overlap helped (+{delta:.2f})"
        elif delta < -1e-9:
            verdict = f"overlap hurt ({delta:.2f})"
        else:
            verdict = "overlap changed nothing"
        lines.append(
            f"{splitter}/{size}: hit-rate {a.hit_rate:.2f} at overlap {lo} vs "
            f"{b.hit_rate:.2f} at overlap {hi}  ->  {verdict}"
        )

    if helped:
        summary = (
            f"Overlap ({lo} vs {hi}) helped for at least one splitter/size on this corpus. "
            "The per-family breakdown below shows exactly where, and where it bought nothing."
        )
    else:
        summary = (
            f"Overlap ({lo} vs {hi}) bought nothing on this corpus: every splitter/size pair "
            "scored the same hit-rate with and without it."
        )
    return OverlapFinding(summary, helped, lines)
