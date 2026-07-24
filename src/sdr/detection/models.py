from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DocumentType(str, Enum):
    PDF_TEXT = "pdf_text"
    PDF_SCANNED = "pdf_scanned"
    DOCX = "docx"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectionResult:
    doc_type: DocumentType
    confidence: float
    signals: tuple[str, ...]
