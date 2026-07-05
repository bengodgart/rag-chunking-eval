"""The three chunk splitters the sweep compares.

All three take ``(text, size, overlap)`` in characters and return a list of
chunk strings. They differ only in where they are allowed to cut:

- ``fixed``     cuts blindly every ``size`` characters (a sliding window).
                It has no idea where a sentence begins or ends, so a chunk
                boundary can land in the middle of the answer.
- ``sentence``  packs whole sentences up to ``size`` and never splits inside a
                sentence, so an answer that lives in one sentence is never cut.
- ``recursive`` splits on the coarsest separator that keeps a piece under
                ``size`` (paragraph, then line, then sentence, then word, then
                character), a simplified version of the common recursive
                character splitter.

``overlap`` is how many characters (fixed/recursive) or trailing characters of
sentence text (sentence) are repeated at the start of the next chunk.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_END.split(text.strip())]
    return [s for s in parts if s]


def fixed_size(text: str, size: int, overlap: int) -> List[str]:
    """Blind sliding window: cut every ``size`` chars, stepping by ``size - overlap``."""
    _validate(size, overlap)
    step = size - overlap
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += step
    return chunks


def sentence(text: str, size: int, overlap: int) -> List[str]:
    """Pack whole sentences up to ``size``; never cut inside a sentence.

    A single sentence longer than ``size`` becomes its own oversized chunk
    rather than being split, which is exactly what preserves an answer that
    lives inside that sentence. ``overlap`` carries whole trailing sentences
    (up to ``overlap`` chars) into the next chunk.
    """
    _validate(size, overlap)
    sents = _sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            carry: List[str] = []
            clen = 0
            for t in reversed(cur):
                if clen + len(t) + 1 <= overlap:
                    carry.insert(0, t)
                    clen += len(t) + 1
                else:
                    break
            cur = list(carry)
            cur_len = sum(len(t) + 1 for t in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]


def _recursive_split(text: str, size: int, seps: List[str]) -> List[str]:
    if len(text) <= size:
        return [text] if text else []
    for i, sep in enumerate(seps):
        if sep == "":
            return [text[j : j + size] for j in range(0, len(text), size)]
        if sep in text:
            parts = text.split(sep)
            out: List[str] = []
            for idx, part in enumerate(parts):
                piece = part + (sep if idx < len(parts) - 1 else "")
                if not piece:
                    continue
                if len(piece) <= size:
                    out.append(piece)
                else:
                    out.extend(_recursive_split(piece, size, seps[i + 1 :]))
            return out
    return [text]


def recursive(text: str, size: int, overlap: int) -> List[str]:
    """Recursive character splitter: coarsest separator that fits, then merge."""
    _validate(size, overlap)
    atoms = _recursive_split(text, size, _SEPARATORS)
    chunks: List[str] = []
    cur = ""
    for piece in atoms:
        if cur and len(cur) + len(piece) > size:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap > 0 else ""
        cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")


SPLITTERS: Dict[str, Callable[[str, int, int], List[str]]] = {
    "fixed": fixed_size,
    "sentence": sentence,
    "recursive": recursive,
}
