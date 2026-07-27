from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source_document: str
    page: int | None
    section_path: tuple[str, ...]
    char_start: int
    char_end: int
    from_table: bool
    chunk_index: int
    strategy: str
