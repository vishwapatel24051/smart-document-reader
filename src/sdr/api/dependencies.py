from __future__ import annotations

from collections.abc import Iterator

import psycopg

from sdr.storage import get_connection


def get_db() -> Iterator[psycopg.Connection]:
    """A connection per request. Opening a fresh connection each call is
    wasteful at high request volume, but fine at this project's scale -
    same reasoning as sdr.storage.db.get_connection()'s own docstring.
    """
    with get_connection() as conn:
        yield conn
