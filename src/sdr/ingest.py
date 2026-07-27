from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg

from sdr.chunking import chunk_structure_aware
from sdr.embedding import embed_texts
from sdr.extraction import extract
from sdr.storage import repository


@dataclass(frozen=True)
class IngestOutcome:
    source_path: str
    status: str  # "indexed" | "skipped_unchanged" | "failed"
    document_id: int | None
    chunks_indexed: int
    reason: str | None = None


def compute_content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_document(path: Path, conn: psycopg.Connection) -> IngestOutcome:
    """Detect, extract, chunk, embed, and store one document.

    Skips entirely (no extraction/chunking/embedding/writes) if the file's
    content hash matches what's already stored - the incremental-indexing
    guarantee. A changed document only ever replaces its own chunk rows
    (see repository.replace_chunks); other documents are never touched.
    """
    source_path = str(path)

    try:
        content_hash = compute_content_hash(path)
    except OSError as exc:
        return IngestOutcome(source_path, "failed", None, 0, f"read_error: {exc}")

    if repository.get_stored_content_hash(conn, source_path) == content_hash:
        return IngestOutcome(source_path, "skipped_unchanged", None, 0)

    document = extract(path)
    if not document.blocks:
        return IngestOutcome(
            source_path, "failed", None, 0, f"extraction_failed: {document.quality.failures}"
        )

    chunks = chunk_structure_aware(document)
    embeddings = embed_texts([c.text for c in chunks])

    document_id = repository.upsert_document(
        conn, source_path, document.doc_type.value, content_hash, document.quality
    )
    chunks_indexed = repository.replace_chunks(conn, document_id, chunks, embeddings)

    return IngestOutcome(source_path, "indexed", document_id, chunks_indexed)
