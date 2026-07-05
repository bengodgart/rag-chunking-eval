# chunkeval retrieval report

**Corpus:** `data/corpus` (9 docs)  
**Questions:** `data/questions.jsonl` (10)  
**Metric:** hit-rate@5 / MRR@5  
**Retrieval:** hand-rolled BM25 (k1=1.5, b=0.75), in-memory, offline

**Winning config:** `recursive/256/64` with hit-rate@5 = **0.90**, MRR@5 = **0.820**

## Config sweep

| Config (splitter/size/overlap) | Chunks | hit-rate@5 | MRR@5 |
|---|---:|---:|---:|
| `recursive/256/64` **(winner)** | 39 | 0.90 | 0.820 |
| `fixed/256/64` | 27 | 0.90 | 0.775 |
| `sentence/256/0` | 29 | 0.90 | 0.720 |
| `sentence/256/64` | 29 | 0.90 | 0.720 |
| `recursive/256/0` | 29 | 0.90 | 0.720 |
| `sentence/128/0` | 45 | 0.80 | 0.800 |
| `sentence/128/64` | 45 | 0.80 | 0.800 |
| `recursive/128/0` | 56 | 0.80 | 0.725 |
| `recursive/128/64` | 63 | 0.70 | 0.600 |
| `fixed/128/64` | 78 | 0.70 | 0.517 |
| `fixed/256/0` | 27 | 0.60 | 0.550 |
| `fixed/128/0` | 45 | 0.50 | 0.450 |

## Overlap finding

Overlap (0 vs 64) helped for at least one splitter/size on this corpus. The per-family breakdown below shows exactly where, and where it bought nothing.

- fixed/128: hit-rate 0.50 at overlap 0 vs 0.70 at overlap 64  ->  overlap helped (+0.20)
- fixed/256: hit-rate 0.60 at overlap 0 vs 0.90 at overlap 64  ->  overlap helped (+0.30)
- recursive/128: hit-rate 0.80 at overlap 0 vs 0.70 at overlap 64  ->  overlap hurt (-0.10)
- recursive/256: hit-rate 0.90 at overlap 0 vs 0.90 at overlap 64  ->  overlap changed nothing
- sentence/128: hit-rate 0.80 at overlap 0 vs 0.80 at overlap 64  ->  overlap changed nothing
- sentence/256: hit-rate 0.90 at overlap 0 vs 0.90 at overlap 64  ->  overlap changed nothing

## How to read this

- **hit-rate@5**: fraction of questions whose gold chunk (right document and, when the question carries an answer span, a chunk actually containing that span) landed in the top 5 retrieved chunks.
- **MRR@5**: mean of 1/rank of the first relevant chunk, counted 0 when none was in the top 5.
- Retrieval is a hand-rolled in-memory BM25 index. No embedding API, no vector database, no network call.