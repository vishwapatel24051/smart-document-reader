from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import TYPE_CHECKING

from sdr.config import get_settings

from .models import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache
def _get_reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().rerank_model)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score candidates with a cross-encoder.

    A cross-encoder sees the query and each chunk together, unlike dense/
    lexical retrieval which score them independently - typically more
    accurate but too slow to run over a whole corpus, hence a second-stage
    reranker over a small candidate pool rather than a first-stage
    retriever.
    """
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    raw_scores = model.predict(pairs)
    scored = sorted(zip(candidates, raw_scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        dataclasses.replace(chunk, score=float(score), rank=i)
        for i, (chunk, score) in enumerate(scored, start=1)
    ]
