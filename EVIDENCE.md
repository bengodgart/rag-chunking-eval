# Evidence — rag-chunking-eval ship gate

All commands run from `C:\dev\rag-chunking-eval`. Python 3.14.4, pytest 9.1.1, standard library only for the tool itself (no third-party dependency: no `rank-bm25`, no `sentence-transformers`, no embedding API, no vector DB, no network). All runs are against the bundled `data/corpus` + `data/questions.jsonl`.

## 1. pytest suite (full output + exit code)

```
$ python -m pytest tests/ -v

============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Asus PC\AppData\Local\Programs\Python\Python314\python.exe
cachedir: .pytest_cache
rootdir: C:\dev\rag-chunking-eval
collecting ... collected 23 items

tests/test_chunkeval.py::test_first_relevant_at_rank_2_hits_at_k5_with_rr_one_half PASSED [  4%]
tests/test_chunkeval.py::test_relevant_below_the_cutoff_is_a_miss_with_zero_rr PASSED [  8%]
tests/test_chunkeval.py::test_no_relevant_chunk_anywhere_is_a_clean_miss PASSED [ 13%]
tests/test_chunkeval.py::test_aggregate_matches_hand_computed_hit_rate_and_mrr PASSED [ 17%]
tests/test_chunkeval.py::test_span_relevance_requires_span_in_chunk_from_right_doc PASSED [ 21%]
tests/test_chunkeval.py::test_span_match_is_whitespace_insensitive PASSED [ 26%]
tests/test_chunkeval.py::test_no_span_falls_back_to_doc_id_match_only PASSED [ 30%]
tests/test_chunkeval.py::test_bm25_ranks_the_on_topic_chunk_first PASSED [ 34%]
tests/test_chunkeval.py::test_bm25_scores_zero_when_no_query_term_appears PASSED [ 39%]
tests/test_chunkeval.py::test_tokenize_lowercases_and_splits_on_non_alphanumeric PASSED [ 43%]
tests/test_chunkeval.py::test_fixed_size_window_overlaps_by_exactly_the_overlap PASSED [ 47%]
tests/test_chunkeval.py::test_overlap_not_smaller_than_size_is_rejected PASSED [ 52%]
tests/test_chunkeval.py::test_sentence_splitter_never_cuts_inside_a_sentence PASSED [ 56%]
tests/test_chunkeval.py::test_recursive_splitter_keeps_chunks_within_size_when_it_can PASSED [ 60%]
tests/test_chunkeval.py::test_every_named_splitter_is_registered PASSED  [ 65%]
tests/test_chunkeval.py::test_corpus_and_questions_load PASSED           [ 69%]
tests/test_chunkeval.py::test_every_answer_span_is_a_real_substring_of_its_gold_doc PASSED [ 73%]
tests/test_chunkeval.py::test_hit_rate_equals_hits_over_questions_for_every_config PASSED [ 78%]
tests/test_chunkeval.py::test_sweep_winner_reaches_point_nine_on_the_bundled_corpus PASSED [ 82%]
tests/test_chunkeval.py::test_overlap_only_rescues_the_blind_fixed_splitter PASSED [ 86%]
tests/test_chunkeval.py::test_fixed_256_zero_overlap_matches_the_hand_checked_numbers PASSED [ 91%]
tests/test_chunkeval.py::test_sweep_is_deterministic_across_runs PASSED  [ 95%]
tests/test_chunkeval.py::test_build_chunks_tags_every_chunk_with_its_source_doc PASSED [100%]

============================= 23 passed in 0.16s ==============================
```

**Exit code: 0.** 23 passed, 0 failed, 0 skipped.

## 2. CLI sweep against the bundled corpus (full output + exit code)

```
$ python -m chunkeval sweep data/corpus data/questions.jsonl --explain fixed/256/0 --md examples/sample-report.md --html examples/sample-report.html

chunkeval - retrieval quality by chunking config
corpus: data/corpus (9 docs)   questions: data/questions.jsonl (10)
metric: hit-rate@5 / MRR@5   retrieval: hand-rolled BM25 (k1=1.5, b=0.75)

Winning config: recursive/256/64   ->   hit-rate@5 = 0.90, MRR@5 = 0.820

Config sweep (best hit-rate first)
  splitter/size/overlap  chunks    hit@5    MRR@5
  recursive/256/64           39     0.90    0.820  [##################--]  <- winner
  fixed/256/64               27     0.90    0.775  [##################--]
  sentence/256/0             29     0.90    0.720  [##################--]
  sentence/256/64            29     0.90    0.720  [##################--]
  recursive/256/0            29     0.90    0.720  [##################--]
  sentence/128/0             45     0.80    0.800  [################----]
  sentence/128/64            45     0.80    0.800  [################----]
  recursive/128/0            56     0.80    0.725  [################----]
  recursive/128/64           63     0.70    0.600  [##############------]
  fixed/128/64               78     0.70    0.517  [##############------]
  fixed/256/0                27     0.60    0.550  [############--------]
  fixed/128/0                45     0.50    0.450  [##########----------]

Overlap finding (overlap 0 vs the largest overlap, holding splitter + size fixed)
  Overlap (0 vs 64) helped for at least one splitter/size on this corpus. The per-family breakdown below shows exactly where, and where it bought nothing.
    fixed/128: hit-rate 0.50 at overlap 0 vs 0.70 at overlap 64  ->  overlap helped (+0.20)
    fixed/256: hit-rate 0.60 at overlap 0 vs 0.90 at overlap 64  ->  overlap helped (+0.30)
    recursive/128: hit-rate 0.80 at overlap 0 vs 0.70 at overlap 64  ->  overlap hurt (-0.10)
    recursive/256: hit-rate 0.90 at overlap 0 vs 0.90 at overlap 64  ->  overlap changed nothing
    sentence/128: hit-rate 0.80 at overlap 0 vs 0.80 at overlap 64  ->  overlap changed nothing
    sentence/256: hit-rate 0.90 at overlap 0 vs 0.90 at overlap 64  ->  overlap changed nothing

Method notes
  - hit-rate@k: fraction of questions whose gold chunk (right doc AND, when given, containing
    the answer span) appears in the top-k retrieved chunks.
  - MRR@k: mean of 1/rank of the first relevant chunk, 0 if none is in the top-k.
  - retrieval is a hand-rolled in-memory BM25 index; no embedding API, no vector DB, no network.

Per-question detail for fixed/256/0  (k=5, 27 chunks)
  question    first-relevant-rank   hit@5   1/rank
  q01                           6      no    0.000
  q02                           -      no    0.000
  q03                           1     yes    1.000
  q04                           2     yes    0.500
  q05                           1     yes    1.000
  q06                           -      no    0.000
  q07                           1     yes    1.000
  q08                           1     yes    1.000
  q09                           -      no    0.000
  q10                           1     yes    1.000

  hits: 6/10  ->  hit-rate@5 = 6/10 = 0.6000
  sum(1/rank): 5.5000  ->  MRR@5 = 5.5000/10 = 0.5500

wrote HTML report: examples/sample-report.html
wrote Markdown report: examples/sample-report.md
```

**Exit code: 0.** Committed copies of this same run: `examples/sample-report.md`, `examples/sample-report.html`.

## 3. Hand-check of hit-rate@5 and MRR@5 (config fixed/256/0)

The `--explain fixed/256/0` table above gives the rank of the first relevant chunk for each of the 10 questions. I verified those ranks independently by dumping the top-5 retrieved chunks for three of them (right doc? chunk contains the answer span?):

```
$ python -c "...top-5 retrieval trace for q03, q04, q01 on fixed/256/0..."

--- q03: What is the step of the water cycle where rain and snow fall called?
    gold doc=water_cycle  span="fall back as rain or snow in the step known as precipitation"
    rank 1: doc=water_cycle            score=13.775 relevant=True
    rank 2: doc=water_cycle            score=7.592 relevant=False
    ...
--- q04: Over long timescales, what happens to carbon in the ocean?
    gold doc=carbon_cycle  span="buried in ocean sediment and slowly pressed into limestone"
    rank 1: doc=plate_tectonics        score=5.221 relevant=False
    rank 2: doc=carbon_cycle           score=5.151 relevant=True
    ...
--- q01: How does photosynthesis store the energy it captures from sunlight?
    gold doc=photosynthesis  span="locks energy away inside molecules of glucose"
    rank 1: doc=photosynthesis         score=10.107 relevant=False
    rank 2: doc=atmosphere_layers      score=5.585 relevant=False
    ...  (first relevant chunk is at rank 6, outside the top 5)
```

Reading those three by hand:

| Q | first relevant chunk | in top 5? | 1/rank |
|---|---|---|---|
| q03 | rank 1 (a water_cycle chunk that contains the span) | yes | 1/1 = 1.000 |
| q04 | rank 2 (rank 1 is a plate_tectonics distractor sharing "ocean"; the carbon_cycle chunk with the span is rank 2) | yes | 1/2 = 0.500 |
| q01 | rank 6 (photosynthesis IS retrieved at rank 1, but fixed/256/0 cut the span across a chunk boundary so the top chunk does not contain the answer text) | no | 0.000 |

q01 is the mechanism the whole tool exists to catch: the right document was retrieved first, yet the answer was a miss because the blind fixed-size cut split the answer span out of that chunk. Overlap fixes exactly this, which is why `fixed/256` climbs from 0.60 to 0.90 once overlap is added.

Extending the same hand-count to all ten questions (hits at q03, q04, q05, q07, q08, q10):

- **hit-rate@5** = 6 hits / 10 questions = **0.6000**
- **MRR@5** = (1.000 + 0.500 + 1.000 + 1.000 + 1.000 + 1.000) / 10 = 5.5000 / 10 = **0.5500**

Both match the tool's printed `0.60` and `0.550` for `fixed/256/0` exactly. The same equality is pinned in the suite by `test_fixed_256_zero_overlap_matches_the_hand_checked_numbers`, and `test_hit_rate_equals_hits_over_questions_for_every_config` checks `hit_rate == hits / Q` for every one of the 12 configs, not just this one.

## 4. The honest overlap-helped-or-not finding

**Overlap only rescued the blind fixed-size splitter; it bought nothing for the splitters that already cut cleanly.**

- Fixed-size splitter: overlap raised hit-rate@5 from **0.60 to 0.90** at size 256 (`+0.30`) and from 0.50 to 0.70 at size 128 (`+0.20`). The blind window keeps cutting answer spans across boundaries, and overlap patches those cuts back together.
- Sentence splitter: **0.90 at overlap 0 and 0.90 at overlap 64** (size 256). No change. It never cuts inside a sentence, so there is nothing for overlap to rescue.
- Recursive splitter: **0.90 with and without overlap** at size 256; at size 128 overlap actually *hurt* slightly (0.80 -> 0.70) by pushing more near-duplicate chunks into the ranking.

The best config on this corpus, `recursive/256/64` (recall@5 = 0.90, MRR@5 = 0.820), ties on hit-rate with `sentence/256/0` and `recursive/256/0`, both of which reach 0.90 with **zero overlap**. So the defensible reading is: on this corpus, choosing a boundary-aware splitter mattered more than adding overlap, and overlap's entire measured benefit was as a band-aid for the naive splitter. That is consistent with, and a sharper version of, the January 2026 "overlap gave no measurable benefit" result.

## Honest gaps (named, not hidden)

- This is a small, clean, original text corpus (9 docs, 10 questions) built so the effect is legible and hand-checkable in seconds. Real corpora are larger and messier; the same numbers on your own docs are the point of the tool, not these specific values.
- Retrieval is lexical BM25 only in v1. A local-embedding backend and a reranker stage are documented v2 paths in `parking_lot.md`, deliberately left out so v1 stays $0, offline, and dependency-free.
- Span relevance is whitespace-insensitive substring containment. It does not do stemming or paraphrase matching, so an answer that a chunk states in different words than the gold span would not count as a hit; this keeps the metric deterministic and hand-verifiable.
