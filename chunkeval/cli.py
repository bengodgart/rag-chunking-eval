"""Command-line interface for chunkeval.

Usage:
    python -m chunkeval sweep <corpus-dir> <questions.jsonl> [options]

Options:
    --k N               top-k cutoff for hit-rate@k and MRR@k (default 5)
    --sizes A,B,...     chunk sizes to sweep (default 128,256)
    --overlaps A,B,...  overlaps to sweep (default 0,64)
    --splitters ...     splitters to sweep, comma-separated from {fixed,sentence,recursive}
                        (default: all three)
    --explain LABEL     also print the per-question rank table for one config,
                        e.g. --explain fixed/256/0 (traces a hit-rate back to questions)
    --html PATH         also write a self-contained HTML report
    --md PATH           also write a Markdown report
    --quiet             suppress the text report on stdout

Exit code 0 on success, 2 on a usage / IO error. Fully offline: no network
calls, no embedding API, no vector database, nothing written outside the paths
you pass to --html / --md.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .corpus import load_corpus, load_questions
from .report import Report, render_explain, render_html, render_markdown, render_text
from .splitters import SPLITTERS
from .sweep import Config, run_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunkeval",
        description="Measure how chunking configs change retrieval quality (offline, BM25).",
    )
    sub = parser.add_subparsers(dest="command")
    s = sub.add_parser("sweep", help="sweep a grid of chunking configs and report hit-rate@k / MRR@k")
    s.add_argument("corpus", help="path to a folder of .txt documents")
    s.add_argument("questions", help="path to a JSONL of {question, doc_id, answer_span?} pairs")
    s.add_argument("--k", type=int, default=5, help="top-k cutoff (default 5)")
    s.add_argument("--sizes", default="128,256", help="comma-separated chunk sizes (default 128,256)")
    s.add_argument("--overlaps", default="0,64", help="comma-separated overlaps (default 0,64)")
    s.add_argument("--splitters", default="fixed,sentence,recursive", help="comma-separated splitters")
    s.add_argument("--explain", default=None, help="print the per-question detail for one config, e.g. fixed/256/0")
    s.add_argument("--html", default=None, help="write an HTML report to this path")
    s.add_argument("--md", default=None, help="write a Markdown report to this path")
    s.add_argument("--quiet", action="store_true", help="suppress the stdout text report")
    return parser


def _int_list(raw: str, name: str) -> List[int]:
    try:
        return [int(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"--{name} must be comma-separated integers, got {raw!r}") from exc


def _grid(sizes: List[int], overlaps: List[int], splitters: List[str]) -> List[Config]:
    for sp in splitters:
        if sp not in SPLITTERS:
            raise ValueError(f"unknown splitter {sp!r}; choose from {sorted(SPLITTERS)}")
    grid: List[Config] = []
    for sp in splitters:
        for size in sizes:
            for overlap in overlaps:
                if overlap >= size:
                    continue  # an overlap >= size is meaningless; skip rather than error
                grid.append(Config(splitter=sp, size=size, overlap=overlap))
    if not grid:
        raise ValueError("the requested grid is empty (check sizes / overlaps / splitters)")
    return grid


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_config_label(label: str) -> Config:
    parts = label.split("/")
    if len(parts) != 3:
        raise ValueError(f"--explain expects splitter/size/overlap, got {label!r}")
    return Config(splitter=parts[0], size=int(parts[1]), overlap=int(parts[2]))


def run(args) -> int:
    try:
        docs = load_corpus(args.corpus)
        questions = load_questions(args.questions)
        sizes = _int_list(args.sizes, "sizes")
        overlaps = _int_list(args.overlaps, "overlaps")
        splitters = [s.strip() for s in args.splitters.split(",") if s.strip()]
        grid = _grid(sizes, overlaps, splitters)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = run_sweep(docs, questions, grid, args.k)
    report = Report(
        corpus_path=args.corpus,
        questions_path=args.questions,
        docs=docs,
        questions=questions,
        results=results,
        k=args.k,
    )

    if not args.quiet:
        print(render_text(report))

    if args.explain:
        try:
            cfg = _parse_config_label(args.explain)
            print("\n" + render_explain(report, cfg))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.html:
        _ensure_parent(args.html)
        with open(args.html, "w", encoding="utf-8") as handle:
            handle.write(render_html(report))
        if not args.quiet:
            print(f"\nwrote HTML report: {args.html}")
    if args.md:
        _ensure_parent(args.md)
        with open(args.md, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(report))
        if not args.quiet:
            print(f"wrote Markdown report: {args.md}")
    return 0


def main(argv: list = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sweep":
        return run(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
