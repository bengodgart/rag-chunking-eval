"""Load the document corpus and the question/answer pairs.

A document's ``doc_id`` is the file stem (``photosynthesis.txt`` -> ``photosynthesis``).
Questions are a JSONL file, one object per line:

    {"id": "q01", "question": "...", "doc_id": "photosynthesis", "answer_span": "..."}

``answer_span`` is optional. When present it is an exact substring of the gold
document, and retrieval is judged on whether a retrieved chunk actually contains
that span (whitespace-normalized). When absent, retrieval is judged on the
``doc_id`` alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Doc:
    doc_id: str
    text: str


@dataclass
class Question:
    id: str
    question: str
    doc_id: str
    answer_span: Optional[str] = None


def load_corpus(path: str) -> List[Doc]:
    """Load every ``*.txt`` file under ``path`` as a document (doc_id = file stem)."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"corpus directory not found: {path}")
    docs: List[Doc] = []
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith(".txt"):
            continue
        full = os.path.join(path, name)
        with open(full, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
        if text:
            docs.append(Doc(doc_id=os.path.splitext(name)[0], text=text))
    if not docs:
        raise ValueError(f"no .txt documents found in {path}")
    return docs


def load_questions(path: str) -> List[Question]:
    """Load a JSONL file of question/answer pairs."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"questions file not found: {path}")
    questions: List[Question] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if "question" not in obj or "doc_id" not in obj:
                raise ValueError(f"{path}:{lineno}: each line needs 'question' and 'doc_id'")
            questions.append(
                Question(
                    id=str(obj.get("id", f"q{lineno}")),
                    question=obj["question"],
                    doc_id=obj["doc_id"],
                    answer_span=obj.get("answer_span"),
                )
            )
    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions
