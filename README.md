# rag-chunking-eval

**Pick a chunking strategy from data instead of vibes.** Point this at a folder of documents plus a few question/answer pairs and it sweeps a small grid of chunking configs (chunk size x overlap x splitter), retrieves against each one with a hand-rolled BM25 index, and reports **hit-rate@k and MRR@k per config** so you can see which chunking actually retrieves the answer.

On the bundled corpus, the headline finding is the interesting kind:

```
hit-rate@5: 0.90 at recursive/256/64 (several size-256 configs tie at 0.90 hit-rate; recursive/256/64 wins on MRR)

The blind fixed-size splitter needed overlap to get there: 0.60 with no overlap, 0.90 with 64.
The boundary-aware splitters already hit 0.90 with ZERO overlap.
So on this corpus, overlap only rescued the naive splitter. Choosing a splitter that
respects sentence boundaries mattered more than adding overlap.
```

That echoes the January 2026 result that chunk overlap gave "no measurable benefit" and sharpens it: overlap bought nothing *for a splitter that was already cutting cleanly*. It was a band-aid for the fixed-size splitter's blind cuts. This tool lets you check that on your own corpus instead of taking anyone's word for it, including mine.

## What the report looks like

```
$ python -m chunkeval sweep data/corpus data/questions.jsonl

Winning config: recursive/256/64   ->   hit-rate@5 = 0.90, MRR@5 = 0.820

Config sweep (best hit-rate first)
  splitter/size/overlap  chunks    hit@5    MRR@5
  recursive/256/64           39     0.90    0.820  [##################--]  <- winner
  fixed/256/64               27     0.90    0.775  [##################--]
  sentence/256/0             29     0.90    0.720  [##################--]
  ...
  fixed/256/0                27     0.60    0.550  [############--------]
  fixed/128/0                45     0.50    0.450  [##########----------]

Overlap finding (overlap 0 vs the largest overlap, holding splitter + size fixed)
  Overlap (0 vs 64) helped for at least one splitter/size on this corpus.
    fixed/256: hit-rate 0.60 at overlap 0 vs 0.90 at overlap 64  ->  overlap helped (+0.30)
    sentence/256: hit-rate 0.90 at overlap 0 vs 0.90 at overlap 64  ->  overlap changed nothing
    recursive/256: hit-rate 0.90 at overlap 0 vs 0.90 at overlap 64  ->  overlap changed nothing
```

That is a real run against the bundled corpus, not a cherry-picked number. The full committed report is in [examples/sample-report.md](examples/sample-report.md) and [examples/sample-report.html](examples/sample-report.html).

## Quickstart (30 seconds, 2 commands)

```bash
git clone https://github.com/bengodgart/rag-chunking-eval
cd rag-chunking-eval
python -m chunkeval sweep data/corpus data/questions.jsonl
```

Python 3.9+, standard library only, nothing to install. `pytest` is only needed to run the tests, not the tool.

## Run it on your own docs

```bash
python -m chunkeval sweep path/to/your/docs path/to/your/questions.jsonl --html report.html --md report.md
```

- **Docs**: a folder of `.txt` files. A document's id is its file name without the extension.
- **Questions**: a JSONL file, one object per line:

```json
{"id": "q01", "question": "How does photosynthesis store energy?", "doc_id": "photosynthesis", "answer_span": "locks energy away inside molecules of glucose"}
```

`answer_span` is optional. When you give it, a retrieval only counts as a hit if the retrieved chunk **actually contains the answer text** (whitespace-insensitive), which is the strict, chunking-sensitive measure that this tool exists to compute. When you leave it out, a hit is just retrieving the right `doc_id`.

## The metrics, defined exactly

- **hit-rate@k** = the fraction of questions whose gold chunk (the right document and, if an answer span is given, a chunk that contains it) appears in the top-k retrieved chunks.
- **MRR@k** = the mean over questions of `1 / rank` of the first relevant chunk, counted as `0` when no relevant chunk is in the top-k.

Both are hand-checked against a manual count in [EVIDENCE.md](EVIDENCE.md), and the metric arithmetic is pinned by the test suite.

## How retrieval works (and what it is not)

Retrieval is a **hand-rolled BM25 index in pure standard library** (`chunkeval/bm25.py`): tokenize, count term frequencies, compute IDF, score with the standard Okapi BM25 formula at `k1=1.5, b=0.75`. There is no embedding API, no hosted vector database, no network call, and no third-party dependency, so it runs at `$0` and fully offline. An optional local-embedding backend is a documented v2 path (see [parking_lot.md](parking_lot.md)), deliberately not built into v1 so nothing heavy or paid sneaks in.

The three splitters are the whole point of the comparison:

- **`fixed`**: a blind sliding window that cuts every `size` characters. It has no idea where a sentence ends, so a boundary can land in the middle of the answer.
- **`sentence`**: packs whole sentences up to `size` and never cuts inside one.
- **`recursive`**: splits on the coarsest separator that keeps a piece under `size` (paragraph, line, sentence, word, character), a simplified version of the common recursive character splitter.

## See why a config scored what it did

```bash
python -m chunkeval sweep data/corpus data/questions.jsonl --explain fixed/256/0
```

prints the per-question table behind that config's number: the rank of the first relevant chunk for each question, whether it was a hit, and its reciprocal rank. That is exactly how the hand-check in [EVIDENCE.md](EVIDENCE.md) traces `0.60` and `0.55` back to the ten individual questions.

## Tests

```bash
python -m pytest tests/ -v   # 23 tests, bundled corpus only, no network, no dependency
```

## Why I built it

Every chunking guide gives a different "best" chunk size, and none of them measured it on my documents. A January 2026 study even found chunk overlap gave no measurable benefit, against years of overlap-by-default advice. Existing "chunk lab" tools **visualize** how text splits but do not **measure** whether a split actually retrieves the answer, which is the thing that decides the config. So I built the measurement: a small harness that turns "best chunk size for RAG" from an opinion into a hit-rate table you can reproduce on your own corpus. It is the difference between "I dumped docs into a vector DB" and "I measured retrieval and improved it."

## License

MIT. See [LICENSE](LICENSE). The bundled corpus is original text written for this repo.
