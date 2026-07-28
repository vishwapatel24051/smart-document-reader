from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from .models import RetrievedChunk, chunk_from_row

# websearch_to_tsquery (not to_tsquery/plainto_tsquery): built for taking a
# raw, arbitrary user search string and never raising a syntax error, while
# still supporting "quoted phrases" and -exclusions if a user types them.
#
# ts_rank_cd is Postgres's own ranking function, not the Okapi BM25 formula -
# see the schema.sql note and the README. This module is "lexical search,"
# deliberately not named/labeled as BM25.
_SQL = """
    SELECT c.id, c.document_id, d.source_path, c.text, c.page, c.section_path,
           c.char_start, c.char_end, c.from_table,
           ts_rank_cd(c.tsv, query) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id,
         websearch_to_tsquery('english', %(query_text)s) query
    WHERE c.tsv @@ query
    ORDER BY score DESC
    LIMIT %(k)s
"""


def search_lexical(conn: psycopg.Connection, query: str, k: int = 10) -> list[RetrievedChunk]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_SQL, {"query_text": query, "k": k})
        rows = cur.fetchall()
    return [chunk_from_row(row, rank=i) for i, row in enumerate(rows, start=1)]
