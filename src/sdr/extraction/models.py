from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sdr.detection import DocumentType


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass(frozen=True)
class ExtractedBlock:
    block_type: BlockType
    text: str
    order: int
    section_path: tuple[str, ...] = ()
    page: int | None = None
    from_ocr: bool = False
    table_rows: tuple[tuple[str, ...], ...] | None = None
    caption: str | None = None


@dataclass(frozen=True)
class QualityReport:
    doc_type: DocumentType
    pages_total: int | None = None
    pages_processed: int = 0
    pages_needing_ocr: int = 0
    pages_ocr_recovered: int = 0
    tables_found: int = 0
    chars_recovered: int = 0
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedDocument:
    source_path: str
    doc_type: DocumentType
    blocks: tuple[ExtractedBlock, ...]
    quality: QualityReport


def failed_extraction(path: Path, doc_type: DocumentType, reason: str) -> ExtractedDocument:
    """Build a result for a document that couldn't be extracted at all.

    Extractors never raise - a document that can't be read or parsed comes
    back with empty blocks and the reason recorded in quality.failures
    instead of an exception, so a bad file doesn't abort a batch ingest.
    """
    return ExtractedDocument(
        source_path=str(path),
        doc_type=doc_type,
        blocks=(),
        quality=QualityReport(doc_type=doc_type, failures=(reason,)),
    )
