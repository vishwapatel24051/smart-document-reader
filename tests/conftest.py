from __future__ import annotations

from collections.abc import Iterator

import httpx
import psycopg
import pytest

from sdr.config import get_settings
from sdr.storage import ensure_schema


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
    settings = get_settings()
    try:
        conn = psycopg.connect(settings.database_url, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.database_url} - run `docker compose up -d db` first: {exc}")

    from pgvector.psycopg import register_vector

    # ensure_schema() must run first: it creates the vector extension, and
    # register_vector() needs that type to already exist to look up its OID.
    ensure_schema(conn)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def ollama_ready() -> None:
    settings = get_settings()
    try:
        httpx.get(f"{settings.ollama_host}/api/version", timeout=2.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama not reachable at {settings.ollama_host} - start it first: {exc}")
