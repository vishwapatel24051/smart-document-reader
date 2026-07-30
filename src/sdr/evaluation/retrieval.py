from __future__ import annotations

import dataclasses
from typing import Any

import psycopg
from psycopg.rows import dict_row

from sdr.embedding import embed_texts
from sdr.retrieval import RetrievedChunk

# Deliberately parallel to sdr/retrieval/{dense,lexical,hybrid}.py rather
# than reusing them directly: those query the production chunks/documents
# tables unconditionally, and threading a config_id/table-name parameter
# through that module for this one-off evaluation concern seemed like more
# coupling than the small amount of duplicated SQL here is worth.

_DENSE_SQL = """
    SELECT id, source_path, text, page, section_path, char_start, char_end, from_table,
           1 - (embedding <=> %(query_vector)s::vector) AS score
    FROM eval_chunks
    WHERE config_id = %(config_id)s AND embedding IS NOT NULL
    ORDER BY embedding <=> %(query_vector)s::vector
    LIMIT %(k)s
"""

_LEXICAL_SQL = """
    SELECT id, source_path, text, page, section_path, char_start, char_end, from_table,
           ts_rank_cd(tsv, query) AS score
    FROM eval_chunks, websearch_to_tsquery('english', %(query_text)s) query
    WHERE config_id = %(config_id)s AND tsv @@ query
    ORDER BY score DESC
    LIMIT %(k)s
"""

_DEFAULT_RRF_K = 60
_POOL_MULTIPLIER = 4


def _row_to_chunk(row: dict[str, Any], rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["id"],
        document_id=0,  # unused for eval grading (source_path is what's graded)
        source_document=row["source_path"],
        text=row["text"],
        page=row["page"],
        section_path=tuple(row["section_path"] or ()),
        char_start=row["char_start"],
        char_end=row["char_end"],
        from_table=row["from_table"],
        score=float(row["score"]),
        rank=rank,
    )


def search_dense(conn: psycopg.Connection, config_id: str, query: str, k: int) -> list[RetrievedChunk]:
    [vector] = embed_texts([query])
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_DENSE_SQL, {"query_vector": vector, "k": k, "config_id": config_id})
        rows = cur.fetchall()
    return [_row_to_chunk(r, i) for i, r in enumerate(rows, start=1)]


def search_lexical(conn: psycopg.Connection, config_id: str, query: str, k: int) -> list[RetrievedChunk]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_LEXICAL_SQL, {"query_text": query, "k": k, "config_id": config_id})
        rows = cur.fetchall()
    return [_row_to_chunk(r, i) for i, r in enumerate(rows, start=1)]


def search_hybrid(
    conn: psycopg.Connection, config_id: str, query: str, k: int, rrf_k: int = _DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    pool = max(k * _POOL_MULTIPLIER, k)
    dense_results = search_dense(conn, config_id, query, pool)
    lexical_results = search_lexical(conn, config_id, query, pool)

    fused_scores: dict[int, float] = {}
    best_chunk: dict[int, RetrievedChunk] = {}
    for results in (dense_results, lexical_results):
        for r in results:
            fused_scores[r.chunk_id] = fused_scores.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + r.rank)
            best_chunk.setdefault(r.chunk_id, r)

    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:k]
    return [
        dataclasses.replace(best_chunk[chunk_id], score=fused_scores[chunk_id], rank=i)
        for i, chunk_id in enumerate(ordered_ids, start=1)
    ]


_STRATEGIES = {"dense": search_dense, "lexical": search_lexical, "hybrid": search_hybrid}


def search(conn: psycopg.Connection, config_id: str, query: str, k: int, strategy: str) -> list[RetrievedChunk]:
    search_fn = _STRATEGIES.get(strategy)
    if search_fn is None:
        raise ValueError(f"unknown retrieval strategy: {strategy!r} (expected one of {sorted(_STRATEGIES)})")
    return search_fn(conn, config_id, query, k)
