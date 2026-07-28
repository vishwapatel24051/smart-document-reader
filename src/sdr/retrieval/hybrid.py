from __future__ import annotations

import dataclasses

import psycopg

from .dense import search_dense
from .lexical import search_lexical
from .models import RetrievedChunk

# The standard constant from the original RRF paper (Cormack et al., 2009),
# not tuned against this project's corpus - no sweep has been run.
_DEFAULT_RRF_K = 60
_POOL_MULTIPLIER = 4


def search_hybrid(
    conn: psycopg.Connection, query: str, k: int = 10, rrf_k: int = _DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Fuse dense and lexical results via Reciprocal Rank Fusion.

    RRF combines by rank, not by raw score, which sidesteps having to
    normalize cosine similarity and ts_rank_cd onto a common scale - they
    aren't comparable numbers to begin with.
    """
    pool = max(k * _POOL_MULTIPLIER, k)
    dense_results = search_dense(conn, query, pool)
    lexical_results = search_lexical(conn, query, pool)

    fused_scores: dict[int, float] = {}
    best_chunk: dict[int, RetrievedChunk] = {}
    for results in (dense_results, lexical_results):
        for result in results:
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + result.rank)
            best_chunk.setdefault(result.chunk_id, result)

    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:k]
    return [
        dataclasses.replace(best_chunk[chunk_id], score=fused_scores[chunk_id], rank=i)
        for i, chunk_id in enumerate(ordered_ids, start=1)
    ]
