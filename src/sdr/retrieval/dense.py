from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from sdr.embedding import embed_texts

from .models import RetrievedChunk, chunk_from_row

# 1 - cosine_distance = cosine_similarity, since embeddings are L2-normalized
# (see sdr.embedding.model). Ordering by the raw distance operator (ASC)
# lets Postgres use the chunks_embedding_idx HNSW index; computing score in
# the SELECT list doesn't affect that.
#
# ::vector is required: unlike an INSERT (where the target column's type
# disambiguates the parameter), a bare `<=>` comparison gives psycopg's
# default list adapter no reason to serialize a Python list as anything but
# a plain float8[], which then has no matching <=> overload against vector.
_SQL = """
    SELECT c.id, c.document_id, d.source_path, c.text, c.page, c.section_path,
           c.char_start, c.char_end, c.from_table,
           1 - (c.embedding <=> %(query_vector)s::vector) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> %(query_vector)s::vector
    LIMIT %(k)s
"""


def search_dense(conn: psycopg.Connection, query: str, k: int = 10) -> list[RetrievedChunk]:
    [vector] = embed_texts([query])
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_SQL, {"query_vector": vector, "k": k})
        rows = cur.fetchall()
    return [chunk_from_row(row, rank=i) for i, row in enumerate(rows, start=1)]
