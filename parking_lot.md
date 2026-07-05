# Parking lot: rag-chunking-eval

Ideas that surfaced during the v1 build. NOT in v1 scope.

- **A reranker stage and its measured lift** - retrieve a wider top-n with BM25, rerank, and report the change in hit-rate@k. The natural v2, and the honest way to show a reranker earns its keep. Different pipeline, out of v1.
- **Local sentence-transformer embedding mode** - a second retrieval backend to sit next to BM25 as a measured comparison, gated OFF by default so the tool stays $0 and offline. Only worth it if it can stay dependency-light; a heavy or paid dependency is a hard no (see the build standard in EVIDENCE.md).
- **Wire hit-rate into CI as a gate** - fail the build when a chunking or corpus change drops recall below a threshold, the retrieval analogue of a coverage gate. Cross-links to the eval-dashboard brief.
- **Larger and messier corpora** - PDFs, HTML, code, mixed languages. v1 bundles a tiny clean text corpus on purpose so the demo runs in seconds and the finding is hand-checkable; real-world loaders are a different scope.
- **More splitters** - token-based (tiktoken), markdown-aware, semantic. v1 keeps three deliberately contrasting ones (blind, sentence-aware, recursive) so the finding is legible, not a wall of near-identical rows.
- **Per-question CSV / JSON export** - machine-readable output for downstream analysis, beyond the `--explain` trace. Nice-to-have, not requested for v1.

Product-creep tripwire (doctrine T11): accounts, a hosted service, uploading anyone's corpus, or an end-to-end RAG chat app means it has stopped being a measurement and become an app. Stop and park.
