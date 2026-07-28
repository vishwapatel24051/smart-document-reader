from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    source_document: str
    text: str
    page: int | None
    section_path: tuple[str, ...]
    char_start: int
    char_end: int
    from_table: bool
    score: float
    rank: int


def chunk_from_row(row: dict[str, Any], rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["id"],
        document_id=row["document_id"],
        source_document=row["source_path"],
        text=row["text"],
        page=row["page"],
        section_path=tuple(row["section_path"] or ()),
        char_start=row["char_start"],
        char_end=row["char_end"],
        from_table=row["from_table"],
        score=float(row["score"]),
        rank=rank,
    )
