from __future__ import annotations

import httpx
import psycopg
from fastapi import APIRouter, Depends

from sdr.config import get_settings

from ..dependencies import get_db
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(conn: psycopg.Connection = Depends(get_db)) -> HealthResponse:
    postgres_ok = False
    pgvector_ok = False
    try:
        conn.execute("SELECT 1")
        postgres_ok = True
        row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
        pgvector_ok = row is not None
    except psycopg.Error:
        pass

    ollama_ok = False
    try:
        settings = get_settings()
        response = httpx.get(f"{settings.ollama_host}/api/version", timeout=2.0)
        ollama_ok = response.status_code == 200
    except httpx.HTTPError:
        pass

    status = "ok" if postgres_ok and pgvector_ok else "degraded"
    return HealthResponse(status=status, postgres=postgres_ok, pgvector=pgvector_ok, ollama=ollama_ok)
