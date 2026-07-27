from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from sdr.config import get_settings

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """A single autocommit connection, schema-ensured and with the pgvector
    type adapter registered. Each ingest/query call is small enough not to
    need connection pooling at this project's scale.

    ensure_schema() must run before register_vector(): the vector type
    adapter looks up the type's OID in the database, which only exists
    once CREATE EXTENSION has run.
    """
    conn = psycopg.connect(get_settings().database_url, autocommit=True)
    try:
        ensure_schema(conn)
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the vector extension, tables, and indexes if they don't
    already exist. Safe to call on every startup - every statement in
    schema.sql is idempotent (IF NOT EXISTS / CREATE TABLE IF NOT EXISTS).
    """
    conn.execute(_SCHEMA_PATH.read_text())
