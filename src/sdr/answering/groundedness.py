from __future__ import annotations

import re

from sdr.retrieval import RetrievedChunk

from .models import AnswerSentence, Citation

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD_RE = re.compile(r"[a-z0-9]+")

_DEFAULT_GROUNDEDNESS_THRESHOLD = 0.4
_MAX_CITATIONS_PER_SENTENCE = 2


def split_sentences(text: str) -> list[str]:
    """A regex sentence splitter, not a real NLP sentence boundary model -
    it will mishandle abbreviations, decimals, etc. Good enough to break a
    short generated answer into citable units, nothing more."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def score_sentence(sentence: str, chunk_text: str) -> float:
    """Fraction of the sentence's distinct words that also appear in the
    chunk's text.

    This is a cheap lexical-overlap heuristic, NOT semantic entailment and
    NOT verified hallucination detection. A sentence can score high while
    asserting something the chunk doesn't actually say (e.g. negating it,
    or combining two unrelated numbers that both appear in the chunk), and
    score low while being a faithful rewording that just uses different
    words. Treat "grounded" as "shares vocabulary with a retrieved
    passage," not as "is factually supported by it."
    """
    sentence_words = _words(sentence)
    if not sentence_words:
        return 0.0
    return len(sentence_words & _words(chunk_text)) / len(sentence_words)


def ground_sentence(
    sentence: str,
    chunks: list[RetrievedChunk],
    threshold: float = _DEFAULT_GROUNDEDNESS_THRESHOLD,
) -> AnswerSentence:
    """Attach citations to a sentence for every retrieved chunk whose
    lexical overlap with it clears `threshold`, capped at the top
    _MAX_CITATIONS_PER_SENTENCE. `grounded` is just "at least one citation
    survived" - see score_sentence() for what that does and doesn't mean.
    """
    scored = sorted(
        ((score_sentence(sentence, c.text), c) for c in chunks),
        key=lambda pair: pair[0],
        reverse=True,
    )
    citations = tuple(
        Citation(document=c.source_document, page=c.page, char_start=c.char_start, char_end=c.char_end, chunk_id=c.chunk_id)
        for score, c in scored[:_MAX_CITATIONS_PER_SENTENCE]
        if score >= threshold
    )
    best_score = scored[0][0] if scored else 0.0
    return AnswerSentence(text=sentence, citations=citations, grounded=len(citations) > 0, support_score=best_score)
