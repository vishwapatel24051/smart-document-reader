from __future__ import annotations

from pathlib import Path

import psycopg
from fastapi import APIRouter, Depends, UploadFile

from sdr.config import get_settings
from sdr.ingest import ingest_document

from ..dependencies import get_db
from ..schemas import IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/documents", response_model=IngestResponse)
async def upload_document(file: UploadFile, conn: psycopg.Connection = Depends(get_db)) -> IngestResponse:
    """Saves the upload to disk under a stable path (by filename) and
    ingests it. Re-uploading a file with the same name and unchanged
    content is a no-op (status: "skipped_unchanged") - incremental
    indexing works the same way here as it does for any other caller of
    sdr.ingest.ingest_document.
    """
    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    assert file.filename, "uploaded file must have a filename"
    dest = upload_dir / Path(file.filename).name  # .name strips any path components
    dest.write_bytes(await file.read())

    outcome = ingest_document(dest, conn)
    return IngestResponse(
        source_path=outcome.source_path,
        status=outcome.status,
        document_id=outcome.document_id,
        chunks_indexed=outcome.chunks_indexed,
        reason=outcome.reason,
    )
