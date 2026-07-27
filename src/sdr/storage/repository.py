from __future__ import annotations

import psycopg

from sdr.chunking import Chunk
from sdr.extraction import QualityReport


def get_stored_content_hash(conn: psycopg.Connection, source_path: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM documents WHERE source_path = %s", (source_path,)
    ).fetchone()
    return row[0] if row else None


def upsert_document(
    conn: psycopg.Connection,
    source_path: str,
    doc_type: str,
    content_hash: str,
    quality: QualityReport,
) -> int:
    row = conn.execute(
        """
        INSERT INTO documents (
            source_path, doc_type, content_hash, pages_total, pages_processed,
            pages_needing_ocr, pages_ocr_recovered, tables_found, chars_recovered,
            extraction_failures, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (source_path) DO UPDATE SET
            doc_type = EXCLUDED.doc_type,
            content_hash = EXCLUDED.content_hash,
            pages_total = EXCLUDED.pages_total,
            pages_processed = EXCLUDED.pages_processed,
            pages_needing_ocr = EXCLUDED.pages_needing_ocr,
            pages_ocr_recovered = EXCLUDED.pages_ocr_recovered,
            tables_found = EXCLUDED.tables_found,
            chars_recovered = EXCLUDED.chars_recovered,
            extraction_failures = EXCLUDED.extraction_failures,
            updated_at = now()
        RETURNING id
        """,
        (
            source_path,
            doc_type,
            content_hash,
            quality.pages_total,
            quality.pages_processed,
            quality.pages_needing_ocr,
            quality.pages_ocr_recovered,
            quality.tables_found,
            quality.chars_recovered,
            list(quality.failures),
        ),
    ).fetchone()
    assert row is not None
    return row[0]


def replace_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Delete this document's existing chunks and insert the new set.

    Scoped to a single document_id, so re-ingesting one changed document
    never touches any other document's chunk rows - that's the whole of
    "incremental indexing" at the storage layer.
    """
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cur.execute(
                """
                INSERT INTO chunks (
                    document_id, chunk_index, strategy, text, page, section_path,
                    char_start, char_end, from_table, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    chunk.chunk_index,
                    chunk.strategy,
                    chunk.text,
                    chunk.page,
                    list(chunk.section_path),
                    chunk.char_start,
                    chunk.char_end,
                    chunk.from_table,
                    embedding,
                ),
            )
    return len(chunks)


def delete_document(conn: psycopg.Connection, source_path: str) -> bool:
    row = conn.execute(
        "DELETE FROM documents WHERE source_path = %s RETURNING id", (source_path,)
    ).fetchone()
    return row is not None


def count_chunks(conn: psycopg.Connection, document_id: int | None = None) -> int:
    if document_id is None:
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    else:
        row = conn.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,)).fetchone()
    return row[0] if row else 0
