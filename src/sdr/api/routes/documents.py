from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from sdr.storage import repository

from ..dependencies import get_db
from ..schemas import DocumentSummary

router = APIRouter(tags=["documents"])


def _to_summary(row: dict) -> DocumentSummary:
    return DocumentSummary(
        id=row["id"],
        source_path=row["source_path"],
        doc_type=row["doc_type"],
        content_hash=row["content_hash"],
        pages_total=row["pages_total"],
        pages_processed=row["pages_processed"],
        pages_needing_ocr=row["pages_needing_ocr"],
        pages_ocr_recovered=row["pages_ocr_recovered"],
        tables_found=row["tables_found"],
        chars_recovered=row["chars_recovered"],
        extraction_failures=row["extraction_failures"] or [],
        ingested_at=row["ingested_at"],
        updated_at=row["updated_at"],
    )


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(conn: psycopg.Connection = Depends(get_db)) -> list[DocumentSummary]:
    return [_to_summary(row) for row in repository.list_documents(conn)]


@router.get("/documents/{document_id}", response_model=DocumentSummary)
def get_document(document_id: int, conn: psycopg.Connection = Depends(get_db)) -> DocumentSummary:
    row = repository.get_document(conn, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no document with id {document_id}")
    return _to_summary(row)
