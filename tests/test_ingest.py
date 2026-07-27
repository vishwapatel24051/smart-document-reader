from __future__ import annotations

from sdr.ingest import ingest_document
from sdr.storage import repository
from helpers import requires_slow_tests


@requires_slow_tests
def test_ingest_then_reingest_unchanged_is_skipped(pg_conn, tmp_path) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("This is a test document.\n\nIt has two paragraphs.")
    repository.delete_document(pg_conn, str(source))

    first = ingest_document(source, pg_conn)
    assert first.status == "indexed"
    assert first.chunks_indexed > 0

    second = ingest_document(source, pg_conn)
    assert second.status == "skipped_unchanged"
    assert second.chunks_indexed == 0

    repository.delete_document(pg_conn, str(source))


@requires_slow_tests
def test_reingesting_a_changed_document_does_not_touch_other_documents(pg_conn, tmp_path) -> None:
    doc_a = tmp_path / "a.txt"
    doc_b = tmp_path / "b.txt"
    doc_a.write_text("Document A, version one.")
    doc_b.write_text("Document B, unrelated content.")
    for p in (doc_a, doc_b):
        repository.delete_document(pg_conn, str(p))

    outcome_a1 = ingest_document(doc_a, pg_conn)
    outcome_b = ingest_document(doc_b, pg_conn)
    assert outcome_a1.status == "indexed"
    assert outcome_b.status == "indexed"

    b_chunk_count_before = repository.count_chunks(pg_conn, outcome_b.document_id)

    doc_a.write_text("Document A, version TWO - completely different content now.")
    outcome_a2 = ingest_document(doc_a, pg_conn)
    assert outcome_a2.status == "indexed"
    assert outcome_a2.document_id == outcome_a1.document_id

    assert repository.count_chunks(pg_conn, outcome_b.document_id) == b_chunk_count_before

    repository.delete_document(pg_conn, str(doc_a))
    repository.delete_document(pg_conn, str(doc_b))


@requires_slow_tests
def test_ingest_of_unextractable_file_does_not_write_to_storage(pg_conn, tmp_path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    repository.delete_document(pg_conn, str(empty))

    outcome = ingest_document(empty, pg_conn)
    assert outcome.status == "failed"
    assert repository.get_stored_content_hash(pg_conn, str(empty)) is None
