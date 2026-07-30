from __future__ import annotations

from pathlib import Path

import psycopg

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def ensure_eval_schema(conn: psycopg.Connection) -> None:
    conn.execute(_SCHEMA_PATH.read_text())
