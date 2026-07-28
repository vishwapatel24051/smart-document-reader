from __future__ import annotations

import psycopg

from sdr.config import get_settings

from .dense import search_dense
from .hybrid import search_hybrid
from .lexical import search_lexical
from .models import RetrievedChunk
from .rerank import rerank as rerank_candidates

_STRATEGIES = {
    "dense": search_dense,
    "lexical": search_lexical,
    "hybrid": search_hybrid,
}

# How much bigger a candidate pool to pull before reranking narrows it back
# down to k. Not tuned - just wide enough that reranking has real
# candidates to reorder.
_RERANK_POOL_MULTIPLIER = 3


def search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    strategy: str | None = None,
    rerank: bool | None = None,
) -> list[RetrievedChunk]:
    """The one interface dense, lexical, and hybrid retrieval sit behind,
    swappable by config (settings.retrieval_strategy / .retrieval_rerank)
    or by explicit argument - the latter is what Phase 7's benchmark matrix
    uses to run the same query through every configuration.
    """
    settings = get_settings()
    resolved_strategy = strategy or settings.retrieval_strategy
    use_rerank = settings.retrieval_rerank if rerank is None else rerank

    search_fn = _STRATEGIES.get(resolved_strategy)
    if search_fn is None:
        raise ValueError(f"unknown retrieval strategy: {resolved_strategy!r} (expected one of {sorted(_STRATEGIES)})")

    pool_k = k * _RERANK_POOL_MULTIPLIER if use_rerank else k
    results = search_fn(conn, query, pool_k)

    if use_rerank:
        results = rerank_candidates(query, results, k)

    return results
