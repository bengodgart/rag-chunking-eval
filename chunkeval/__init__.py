"""chunkeval - measure how a chunking config changes retrieval quality.

A tiny, dependency-free harness that sweeps a grid of chunking configs over a
document folder plus a set of question/answer pairs, builds a hand-rolled BM25
index per config, retrieves top-k per question, and reports hit-rate@k and MRR@k
so you can pick a chunking strategy from data instead of guesswork.

Offline, standard library only. No embedding API, no vector database.
"""

__version__ = "0.1.0"
