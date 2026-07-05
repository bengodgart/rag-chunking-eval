"""Tests for chunkeval. Runs against the bundled corpus only, no network,
no external corpus, no third-party dependency.

Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkeval.bm25 import BM25Index, tokenize
from chunkeval.corpus import load_corpus, load_questions
from chunkeval.metrics import aggregate, evaluate_question, is_relevant
from chunkeval.splitters import SPLITTERS, fixed_size, recursive, sentence
from chunkeval.sweep import (
    Config,
    build_chunks,
    default_grid,
    overlap_finding,
    pick_winner,
    run_sweep,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
QUESTIONS = ROOT / "data" / "questions.jsonl"


# ---------------------------------------------------------------------------
# Metric correctness: hit-rate@k and MRR@k on hand-built ranked-relevance lists
# (no BM25, no corpus - just the arithmetic, verifiable by eye).
# ---------------------------------------------------------------------------

def test_first_relevant_at_rank_2_hits_at_k5_with_rr_one_half():
    # ranking: miss, HIT, miss, hit  -> first relevant at rank 2
    qr = evaluate_question([False, True, False, True], k=5)
    assert qr.first_relevant_rank == 2
    assert qr.hit is True
    assert qr.reciprocal_rank == pytest.approx(0.5)


def test_relevant_below_the_cutoff_is_a_miss_with_zero_rr():
    # first relevant at rank 3, but k=2 -> outside the cutoff -> miss, rr 0
    qr = evaluate_question([False, False, True], k=2)
    assert qr.first_relevant_rank == 3
    assert qr.hit is False
    assert qr.reciprocal_rank == 0.0


def test_no_relevant_chunk_anywhere_is_a_clean_miss():
    qr = evaluate_question([False, False, False], k=5)
    assert qr.first_relevant_rank is None
    assert qr.hit is False
    assert qr.reciprocal_rank == 0.0


def test_aggregate_matches_hand_computed_hit_rate_and_mrr():
    # three questions: ranks 1, 4 (k=5 -> hit, rr .25), and none
    a = evaluate_question([True], k=5)                       # rr 1.0, hit
    b = evaluate_question([False, False, False, True], k=5)  # rr .25, hit
    c = evaluate_question([False], k=5)                      # rr 0, miss
    hit_rate, mrr = aggregate([a, b, c])
    # hits 2/3; mrr = (1.0 + 0.25 + 0.0) / 3
    assert hit_rate == pytest.approx(2 / 3)
    assert mrr == pytest.approx((1.0 + 0.25) / 3)


# ---------------------------------------------------------------------------
# Relevance rule: right doc AND (when given) a chunk that contains the span,
# whitespace-insensitive so a splitter's rejoining does not break the match.
# ---------------------------------------------------------------------------

def test_span_relevance_requires_span_in_chunk_from_right_doc():
    assert is_relevant("the cat sat on the mat", "d1", "d1", "cat sat") is True
    assert is_relevant("the cat sat on the mat", "d2", "d1", "cat sat") is False  # wrong doc
    assert is_relevant("the dog ran", "d1", "d1", "cat sat") is False             # span absent


def test_span_match_is_whitespace_insensitive():
    assert is_relevant("the  cat\n  sat  on", "d1", "d1", "cat sat") is True


def test_no_span_falls_back_to_doc_id_match_only():
    assert is_relevant("anything at all", "d1", "d1", None) is True
    assert is_relevant("anything at all", "d2", "d1", None) is False


# ---------------------------------------------------------------------------
# BM25: a query ranks the on-topic chunk first, and an unseen term scores 0
# ---------------------------------------------------------------------------

def test_bm25_ranks_the_on_topic_chunk_first():
    chunks = [
        "plate tectonics moves the continents across the mantle",
        "photosynthesis turns sunlight into glucose inside the leaf",
        "the water cycle lifts vapour off the ocean into clouds",
    ]
    index = BM25Index(chunks)
    ranked = index.rank("how does photosynthesis make glucose from sunlight")
    assert ranked[0][0] == 1          # the photosynthesis chunk
    assert ranked[0][1] > ranked[1][1]  # and it strictly out-scores the runner-up


def test_bm25_scores_zero_when_no_query_term_appears():
    index = BM25Index(["alpha beta gamma", "delta epsilon zeta"])
    ranked = index.rank("nonexistent vocabulary term")
    assert all(score == 0.0 for _, score in ranked)


def test_tokenize_lowercases_and_splits_on_non_alphanumeric():
    assert tokenize("Rain, snow; and HAIL!") == ["rain", "snow", "and", "hail"]


# ---------------------------------------------------------------------------
# Splitters: the boundary behaviour each one is supposed to have
# ---------------------------------------------------------------------------

def test_fixed_size_window_overlaps_by_exactly_the_overlap():
    chunks = fixed_size("abcdefghij", size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]
    # consecutive chunks share the last/first `overlap` characters
    assert chunks[0][-2:] == chunks[1][:2]


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        fixed_size("abcdef", size=4, overlap=4)
    with pytest.raises(ValueError):
        recursive("abcdef", size=4, overlap=9)


def test_sentence_splitter_never_cuts_inside_a_sentence():
    text = "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu."
    chunks = sentence(text, size=40, overlap=0)
    assert len(chunks) > 1  # size forces more than one chunk
    for s in ["Alpha beta gamma delta.", "Epsilon zeta eta theta.", "Iota kappa lambda mu."]:
        assert any(s in c for c in chunks), f"sentence was cut: {s!r}"


def test_recursive_splitter_keeps_chunks_within_size_when_it_can():
    text = "one two three. four five six. seven eight nine. ten eleven twelve."
    chunks = recursive(text, size=30, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 30 for c in chunks)


def test_every_named_splitter_is_registered():
    assert set(SPLITTERS) == {"fixed", "sentence", "recursive"}


# ---------------------------------------------------------------------------
# End-to-end sweep over the bundled corpus: it runs, reconciles, and produces
# the honest overlap finding (locks the headline numbers).
# ---------------------------------------------------------------------------

def test_corpus_and_questions_load():
    docs = load_corpus(str(CORPUS))
    questions = load_questions(str(QUESTIONS))
    assert len(docs) == 9
    assert len(questions) == 10
    # every question points at a doc that actually exists in the corpus
    doc_ids = {d.doc_id for d in docs}
    assert all(q.doc_id in doc_ids for q in questions)


def test_every_answer_span_is_a_real_substring_of_its_gold_doc():
    docs = {d.doc_id: d.text for d in load_corpus(str(CORPUS))}
    for q in load_questions(str(QUESTIONS)):
        if q.answer_span:
            norm_doc = " ".join(docs[q.doc_id].lower().split())
            norm_span = " ".join(q.answer_span.lower().split())
            assert norm_span in norm_doc, f"{q.id}: span not found in {q.doc_id}"


def test_hit_rate_equals_hits_over_questions_for_every_config():
    docs = load_corpus(str(CORPUS))
    questions = load_questions(str(QUESTIONS))
    for r in run_sweep(docs, questions, default_grid(), k=5):
        hits = sum(1 for qr in r.per_question if qr.hit)
        assert r.hit_rate == pytest.approx(hits / len(questions))


def test_sweep_winner_reaches_point_nine_on_the_bundled_corpus():
    docs = load_corpus(str(CORPUS))
    questions = load_questions(str(QUESTIONS))
    results = run_sweep(docs, questions, default_grid(), k=5)
    winner = pick_winner(results)
    assert winner.hit_rate == pytest.approx(0.9)


def test_overlap_only_rescues_the_blind_fixed_splitter():
    docs = load_corpus(str(CORPUS))
    questions = load_questions(str(QUESTIONS))
    results = run_sweep(docs, questions, default_grid(), k=5)
    by = {r.config: r for r in results}
    # overlap lifts the blind fixed splitter...
    assert by[Config("fixed", 256, 64)].hit_rate > by[Config("fixed", 256, 0)].hit_rate
    # ...but the boundary-aware sentence splitter already scores its best with none
    assert by[Config("sentence", 256, 64)].hit_rate == by[Config("sentence", 256, 0)].hit_rate
    finding = overlap_finding(results)
    assert finding.helped_anywhere is True


def test_fixed_256_zero_overlap_matches_the_hand_checked_numbers():
    # the exact config hand-verified in EVIDENCE.md: 6/10 hits, MRR 0.55
    docs = load_corpus(str(CORPUS))
    questions = load_questions(str(QUESTIONS))
    results = run_sweep(docs, questions, [Config("fixed", 256, 0)], k=5)
    r = results[0]
    assert r.hit_rate == pytest.approx(0.6)
    assert r.mrr == pytest.approx(0.55)


def test_sweep_is_deterministic_across_runs():
    docs = load_corpus(str(CORPUS))
    questions = load_questions(str(QUESTIONS))
    a = run_sweep(docs, questions, default_grid(), k=5)
    b = run_sweep(docs, questions, default_grid(), k=5)
    assert [(r.config, r.hit_rate, r.mrr) for r in a] == [(r.config, r.hit_rate, r.mrr) for r in b]


def test_build_chunks_tags_every_chunk_with_its_source_doc():
    docs = load_corpus(str(CORPUS))
    chunks = build_chunks(docs, Config("sentence", 256, 0))
    doc_ids = {d.doc_id for d in docs}
    assert chunks and all(c.doc_id in doc_ids for c in chunks)
