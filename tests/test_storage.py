from __future__ import annotations

from sdr.chunking import Chunk
from sdr.detection import DocumentType
from sdr.extraction import QualityReport
from sdr.storage import ensure_schema, repository


def test_ensure_schema_is_idempotent(pg_conn) -> None:
    ensure_schema(pg_conn)  # the fixture already applied it once
    ensure_schema(pg_conn)


def test_pgvector_extension_is_installed(pg_conn) -> None:
    row = pg_conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None


def test_expected_tables_exist(pg_conn) -> None:
    rows = pg_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {"documents", "chunks"} <= names


def test_document_upsert_and_lookup_roundtrip(pg_conn) -> None:
    source_path = "/tmp/sdr_test_storage_roundtrip.txt"
    pg_conn.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))

    quality = QualityReport(doc_type=DocumentType.PLAIN_TEXT, pages_processed=1, chars_recovered=42)
    doc_id = repository.upsert_document(pg_conn, source_path, "plain_text", "hash-abc", quality)
    assert isinstance(doc_id, int)
    assert repository.get_stored_content_hash(pg_conn, source_path) == "hash-abc"

    # Re-upsert with a new hash must update the same row, not duplicate it.
    doc_id_2 = repository.upsert_document(pg_conn, source_path, "plain_text", "hash-def", quality)
    assert doc_id_2 == doc_id
    assert repository.get_stored_content_hash(pg_conn, source_path) == "hash-def"

    assert repository.delete_document(pg_conn, source_path) is True
    assert repository.get_stored_content_hash(pg_conn, source_path) is None


def _make_chunk(doc_path: str, idx: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        source_document=doc_path,
        page=None,
        section_path=(),
        char_start=0,
        char_end=len(text),
        from_table=False,
        chunk_index=idx,
        strategy="structure_aware",
    )


def test_replace_chunks_only_touches_its_own_document(pg_conn) -> None:
    quality = QualityReport(doc_type=DocumentType.PLAIN_TEXT)
    path_a = "/tmp/sdr_test_storage_doc_a.txt"
    path_b = "/tmp/sdr_test_storage_doc_b.txt"
    for p in (path_a, path_b):
        pg_conn.execute("DELETE FROM documents WHERE source_path = %s", (p,))

    doc_a = repository.upsert_document(pg_conn, path_a, "plain_text", "hash-a1", quality)
    doc_b = repository.upsert_document(pg_conn, path_b, "plain_text", "hash-b1", quality)

    chunks_a = [_make_chunk(path_a, 0, "chunk a0"), _make_chunk(path_a, 1, "chunk a1")]
    chunks_b = [_make_chunk(path_b, 0, "chunk b0")]

    repository.replace_chunks(pg_conn, doc_a, chunks_a, [[0.0] * 384] * len(chunks_a))
    repository.replace_chunks(pg_conn, doc_b, chunks_b, [[0.0] * 384] * len(chunks_b))

    assert repository.count_chunks(pg_conn, doc_a) == 2
    assert repository.count_chunks(pg_conn, doc_b) == 1

    # Re-indexing doc_a with a different chunk set must not touch doc_b.
    new_chunks_a = [_make_chunk(path_a, 0, "updated a0")]
    repository.replace_chunks(pg_conn, doc_a, new_chunks_a, [[0.0] * 384])

    assert repository.count_chunks(pg_conn, doc_a) == 1
    assert repository.count_chunks(pg_conn, doc_b) == 1

    repository.delete_document(pg_conn, path_a)
    repository.delete_document(pg_conn, path_b)
