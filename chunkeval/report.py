"""Render a sweep result to text, Markdown, or self-contained dark HTML."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import List

from .corpus import Doc, Question
from .sweep import Config, ConfigResult, OverlapFinding, overlap_finding, pick_winner

BAR_WIDTH = 20


@dataclass
class Report:
    corpus_path: str
    questions_path: str
    docs: List[Doc]
    questions: List[Question]
    results: List[ConfigResult]
    k: int


def _bar(share: float) -> str:
    share = max(0.0, min(1.0, share))
    filled = int(round(share * BAR_WIDTH))
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def _ordered(results: List[ConfigResult]) -> List[ConfigResult]:
    return sorted(results, key=lambda r: (-r.hit_rate, -r.mrr, r.num_chunks))


def _headline(report: Report) -> str:
    winner = pick_winner(report.results)
    return (
        f"hit-rate@{report.k}: {winner.hit_rate:.2f} at {winner.config.label()} "
        f"(MRR@{report.k} {winner.mrr:.3f})"
    )


def render_text(report: Report) -> str:
    k = report.k
    winner = pick_winner(report.results)
    finding = overlap_finding(report.results)
    lines: List[str] = []

    lines.append("chunkeval - retrieval quality by chunking config")
    lines.append(
        f"corpus: {report.corpus_path} ({len(report.docs)} docs)   "
        f"questions: {report.questions_path} ({len(report.questions)})"
    )
    lines.append(f"metric: hit-rate@{k} / MRR@{k}   retrieval: hand-rolled BM25 (k1=1.5, b=0.75)")
    lines.append("")
    lines.append(f"Winning config: {winner.config.label()}   ->   hit-rate@{k} = {winner.hit_rate:.2f}, MRR@{k} = {winner.mrr:.3f}")
    lines.append("")

    lines.append("Config sweep (best hit-rate first)")
    lines.append(f"  {'splitter/size/overlap':22} {'chunks':>6}  {'hit@'+str(k):>7}  {'MRR@'+str(k):>7}")
    for r in _ordered(report.results):
        mark = "  <- winner" if r.config == winner.config else ""
        lines.append(
            f"  {r.config.label():22} {r.num_chunks:>6}  {r.hit_rate:>7.2f}  {r.mrr:>7.3f}  "
            f"[{_bar(r.hit_rate)}]{mark}"
        )
    lines.append("")

    lines.append("Overlap finding (overlap 0 vs the largest overlap, holding splitter + size fixed)")
    lines.append(f"  {finding.text}")
    for row in finding.per_family:
        lines.append(f"    {row}")
    lines.append("")

    lines.append("Method notes")
    lines.append("  - hit-rate@k: fraction of questions whose gold chunk (right doc AND, when given, containing")
    lines.append("    the answer span) appears in the top-k retrieved chunks.")
    lines.append("  - MRR@k: mean of 1/rank of the first relevant chunk, 0 if none is in the top-k.")
    lines.append("  - retrieval is a hand-rolled in-memory BM25 index; no embedding API, no vector DB, no network.")
    return "\n".join(lines)


def render_explain(report: Report, config: Config) -> str:
    """Per-question first-relevant-rank table for one config, so a hit-rate / MRR
    number can be traced back to the individual questions behind it."""
    result = next((r for r in report.results if r.config == config), None)
    if result is None:
        raise ValueError(f"config {config.label()} not in the sweep grid")
    k = report.k
    lines: List[str] = []
    lines.append(f"Per-question detail for {config.label()}  (k={k}, {result.num_chunks} chunks)")
    lines.append(f"  {'question':10} {'first-relevant-rank':>20} {'hit@'+str(k):>7} {'1/rank':>8}")
    for qr in result.per_question:
        rank = qr.first_relevant_rank if qr.first_relevant_rank is not None else "-"
        lines.append(
            f"  {qr.question_id:10} {str(rank):>20} {('yes' if qr.hit else 'no'):>7} {qr.reciprocal_rank:>8.3f}"
        )
    hits = sum(1 for qr in result.per_question if qr.hit)
    q = len(result.per_question)
    rr_sum = sum(qr.reciprocal_rank for qr in result.per_question)
    lines.append("")
    lines.append(f"  hits: {hits}/{q}  ->  hit-rate@{k} = {hits}/{q} = {result.hit_rate:.4f}")
    lines.append(f"  sum(1/rank): {rr_sum:.4f}  ->  MRR@{k} = {rr_sum:.4f}/{q} = {result.mrr:.4f}")
    return "\n".join(lines)


def render_markdown(report: Report) -> str:
    k = report.k
    winner = pick_winner(report.results)
    finding = overlap_finding(report.results)
    md: List[str] = []

    md.append("# chunkeval retrieval report")
    md.append("")
    md.append(f"**Corpus:** `{report.corpus_path}` ({len(report.docs)} docs)  ")
    md.append(f"**Questions:** `{report.questions_path}` ({len(report.questions)})  ")
    md.append(f"**Metric:** hit-rate@{k} / MRR@{k}  ")
    md.append(f"**Retrieval:** hand-rolled BM25 (k1=1.5, b=0.75), in-memory, offline")
    md.append("")
    md.append(f"**Winning config:** `{winner.config.label()}` with hit-rate@{k} = **{winner.hit_rate:.2f}**, MRR@{k} = **{winner.mrr:.3f}**")
    md.append("")

    md.append("## Config sweep")
    md.append("")
    md.append(f"| Config (splitter/size/overlap) | Chunks | hit-rate@{k} | MRR@{k} |")
    md.append("|---|---:|---:|---:|")
    for r in _ordered(report.results):
        mark = " **(winner)**" if r.config == winner.config else ""
        md.append(f"| `{r.config.label()}`{mark} | {r.num_chunks} | {r.hit_rate:.2f} | {r.mrr:.3f} |")
    md.append("")

    md.append("## Overlap finding")
    md.append("")
    md.append(finding.text)
    md.append("")
    for row in finding.per_family:
        md.append(f"- {row}")
    md.append("")

    md.append("## How to read this")
    md.append("")
    md.append(f"- **hit-rate@{k}**: fraction of questions whose gold chunk (right document and, when the question carries an answer span, a chunk actually containing that span) landed in the top {k} retrieved chunks.")
    md.append(f"- **MRR@{k}**: mean of 1/rank of the first relevant chunk, counted 0 when none was in the top {k}.")
    md.append("- Retrieval is a hand-rolled in-memory BM25 index. No embedding API, no vector database, no network call.")
    return "\n".join(md)


def render_html(report: Report) -> str:
    k = report.k
    winner = pick_winner(report.results)
    finding = overlap_finding(report.results)

    def rows() -> str:
        out = ""
        for r in _ordered(report.results):
            is_win = r.config == winner.config
            cls = " class='win'" if is_win else ""
            tag = " &#9733;" if is_win else ""
            out += (
                f"<tr{cls}><td><code>{html.escape(r.config.label())}</code>{tag}</td>"
                f"<td class='num'>{r.num_chunks}</td>"
                f"<td class='barcell'><div class='bar'><div class='fill' style='width:{r.hit_rate*100:.0f}%'></div></div>"
                f"<span class='pct'>{r.hit_rate:.2f}</span></td>"
                f"<td class='num'>{r.mrr:.3f}</td></tr>"
            )
        return out

    finding_rows = "".join(
        f"<li>{html.escape(row)}</li>" for row in finding.per_family
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>chunkeval retrieval report</title>
<style>
  body{{margin:0;background:#0f172a;color:#e2e8f0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.6;}}
  .wrap{{max-width:840px;margin:0 auto;padding:32px 20px 64px;}}
  h1{{font-size:1.6rem;margin:0 0 2px;}}
  h2{{font-size:1.15rem;border-bottom:1px solid #334155;padding-bottom:.3em;margin:1.8em 0 .6em;color:#fff;}}
  .sub{{color:#94a3b8;font-size:.9rem;}}
  table{{width:100%;border-collapse:collapse;margin:8px 0;font-size:.9rem;}}
  th{{text-align:left;color:#94a3b8;font-weight:600;font-size:.82rem;border-bottom:1px solid #334155;padding:6px 10px;}}
  th.num,td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}}
  td{{padding:6px 10px;border-bottom:1px solid #334155;vertical-align:middle;}}
  tr.win td{{background:rgba(34,197,94,.10);}}
  .barcell{{width:38%;}}
  .bar{{display:inline-block;width:calc(100% - 44px);height:12px;background:#243349;border-radius:6px;overflow:hidden;vertical-align:middle;}}
  .fill{{height:100%;background:#38bdf8;}}
  .pct{{margin-left:8px;color:#94a3b8;font-size:.8rem;}}
  .totalbox{{background:#1e293b;border:1px solid #334155;border-left:3px solid #22c55e;border-radius:8px;padding:14px 18px;margin:14px 0;}}
  .totalbox .big{{font-size:1.35rem;font-weight:700;color:#fff;}}
  .findbox{{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px 18px;margin:12px 0;}}
  .findbox ul{{margin:.5em 0 0;padding-left:1.1em;}}
  .findbox li{{color:#cbd5e1;font-size:.9rem;margin:2px 0;font-variant-numeric:tabular-nums;}}
  code{{background:#243349;padding:1px 6px;border-radius:5px;font-size:.85em;color:#cbd5e1;}}
</style></head><body><div class="wrap">
<h1>chunkeval retrieval report</h1>
<p class="sub">corpus <code>{html.escape(report.corpus_path)}</code> ({len(report.docs)} docs) &middot; questions <code>{html.escape(report.questions_path)}</code> ({len(report.questions)})</p>
<p class="sub">metric hit-rate@{k} / MRR@{k} &middot; retrieval: hand-rolled in-memory BM25 (k1=1.5, b=0.75), offline</p>

<div class="totalbox">
  <div class="big">hit-rate@{k}: {winner.hit_rate:.2f} at {html.escape(winner.config.label())}</div>
  <div class="sub" style="margin-top:4px">winning config, MRR@{k} {winner.mrr:.3f}</div>
</div>

<h2>Config sweep</h2>
<table>
<thead><tr><th>Config (splitter/size/overlap)</th><th class="num">Chunks</th><th>hit-rate@{k}</th><th class="num">MRR@{k}</th></tr></thead>
<tbody>{rows()}</tbody>
</table>

<h2>Overlap finding</h2>
<div class="findbox">
<p style="margin:0">{html.escape(finding.text)}</p>
<ul>{finding_rows}</ul>
</div>

<h2>How to read this</h2>
<p class="sub">hit-rate@{k} is the fraction of questions whose gold chunk (the right document and, when the question carries an answer span, a chunk that actually contains that span) landed in the top {k} retrieved chunks. MRR@{k} is the mean of 1/rank of the first relevant chunk, counted 0 when none was in the top {k}. Retrieval is a hand-rolled BM25 index in memory. No embedding API, no vector database, no network call.</p>
<p class="sub" style="margin-top:24px">Generated by chunkeval. Offline, deterministic, standard library only.</p>
</div></body></html>"""
