# PRD — rag-chunking-eval

**One-liner (from brief 10):** A free harness that takes a folder of documents plus a handful of question/answer pairs, tries several chunking configs (size, overlap, splitter), and reports retrieval hit-rate per config, so you pick a chunking strategy with data instead of vibes.

**Usefulness (from brief 10):** Chunking is tuned blind ("the art of chunking is striking the perfect balance"), and a January 2026 study even contradicts the old overlap defaults with a "no measurable benefit" finding. Existing chunk-lab tools visualize splits but do not measure retrieval quality of different configs against the user's own Q&A pairs, which is the thing that actually decides the config. This reframes the piece from a saturated visualizer into a differentiated eval, and hits the RAG rarely-evidenced differentiator directly: a before/after-on-chunking measurement is the senior version of "I dumped docs into a vector DB."

## v1 scope (capped): each item traces to the brief

1. **Inputs**: a docs folder (`.txt` files, doc_id = file stem) plus a JSONL of `{question, doc_id, answer_span?}` pairs. A bundled sample corpus + questions ship in `data/`. (brief v1.1)
   - Code: `chunkeval/corpus.py`.
2. **Sweep a small, explicit grid**: chunk sizes x overlaps x splitter, with the splitter being fixed-size vs sentence vs recursive. Default grid is 3 splitters x {128, 256} x {0, 64} = 12 configs, all overridable from the CLI. (brief v1.2)
   - Code: `chunkeval/splitters.py`, `chunkeval/sweep.py`; flags `--sizes --overlaps --splitters`.
3. **Per config**: build an in-memory index (hand-rolled BM25, no paid embedding, no vector DB), retrieve top-k per question, compute hit-rate@k and MRR. (brief v1.3)
   - Code: `chunkeval/bm25.py`, `chunkeval/metrics.py`, `chunkeval/sweep.py`.
4. **One output**: a config -> hit-rate@k / MRR table with the winning config highlighted, plus a one-line "overlap helped / did not" finding, committed as HTML + md. (brief v1.4)
   - Code: `chunkeval/report.py`; committed to `examples/sample-report.{md,html}`.
5. **README opens with the headline** ("hit-rate@5: 0.90 at recursive/256/64, and overlap only rescued the blind fixed splitter") plus a 30-second run block. (brief v1.5)
   - `README.md`.

## The metric, exactly (brief: "the metric must be correct")

- **hit-rate@k** = fraction of questions whose gold chunk (right `doc_id` AND, when an `answer_span` is given, a chunk that contains that span, whitespace-insensitive) appears in the top-k retrieved chunks.
- **MRR@k** = mean over questions of `1/rank` of the first relevant chunk, `0` if none is in the top-k.
- Both are hand-checked against a manual count in `EVIDENCE.md` and pinned by the test suite (`tests/test_chunkeval.py`).

## Build standards honored (brief: non-negotiable)

- **$0 and fully offline.** Retrieval is a hand-rolled BM25 (`k1=1.5, b=0.75`) in pure standard library. No `rank-bm25`, no `sentence-transformers`, no embedding API, no hosted vector DB, no network call. `dependencies = []` in `pyproject.toml`.
- **Bundled corpus designed so configs genuinely differ.** The corpus + Q&A are built so at least one config visibly beats another (fixed/256 goes 0.60 -> 0.90 with overlap; sentence/recursive already reach 0.90 without it), so the "overlap helped or not" finding is real and not a flat tie.
- **No em-dashes in user-facing copy** (README, posts, HTML, CLI output). The single em-dash in this PRD title is the accepted house convention.

## Non-goals (NOT v1 - expansion paths, parked in `parking_lot.md`)

- A hosted service, accounts, or any corpus upload.
- A paid embedding or vector-DB dependency.
- An end-to-end RAG chat app.
- Reranking (v2), and generation-quality eval (this measures retrieval only, on purpose).

## Demo path (stranger sees value in under 2 minutes)

Clone -> `python -m chunkeval sweep data/corpus data/questions.jsonl` -> read the config -> hit-rate@5 table and the overlap finding. Then `--explain fixed/256/0` to see the per-question ranks behind a number, then point it at their own docs folder + questions JSONL.

## Done when (brief checklist)

- The CLI runs the sweep on the bundled corpus and emits the hit-rate table in seconds, fully offline at $0. (met)
- The winning config's hit-rate matches a hand-checked spot count on 2-3 questions, pasted in `EVIDENCE.md`. (met)
- The report states an honest overlap-helped-or-not finding with the numbers behind it. (met: overlap helped only the fixed splitter)
- README opens with the headline number; copy passes the no-em-dash sweep. (met)
- Repo public, MIT, tests pass. (23 tests pass; publish is the main thread's step)
