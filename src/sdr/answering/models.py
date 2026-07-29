from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    document: str
    page: int | None
    char_start: int
    char_end: int
    chunk_id: int


@dataclass(frozen=True)
class AnswerSentence:
    text: str
    citations: tuple[Citation, ...]
    grounded: bool
    # Lexical-overlap heuristic score (see groundedness.py) - not a
    # confidence probability and not verified hallucination detection.
    support_score: float


@dataclass(frozen=True)
class Answer:
    query: str
    sentences: tuple[AnswerSentence, ...]
    raw_text: str
    retrieved_chunk_ids: tuple[int, ...]
