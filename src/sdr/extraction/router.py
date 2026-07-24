from __future__ import annotations

from pathlib import Path

from sdr.detection import DocumentType, detect_file

from . import docx as docx_extractor
from . import html as html_extractor
from . import pdf as pdf_extractor
from . import text as text_extractor
from .models import ExtractedDocument, QualityReport

_ROUTES = {
    DocumentType.PDF_TEXT: pdf_extractor.extract,
    DocumentType.PDF_SCANNED: pdf_extractor.extract,
    DocumentType.DOCX: docx_extractor.extract,
    DocumentType.HTML: html_extractor.extract,
    DocumentType.PLAIN_TEXT: text_extractor.extract,
}


def extract(path: Path) -> ExtractedDocument:
    """Detect a document's format and route it to the matching extractor.

    Never raises: an unsupported or undetectable format comes back as an
    ExtractedDocument with no blocks and the reason in quality.failures,
    the same convention every extractor follows for its own failure modes.
    """
    detection = detect_file(path)
    extractor = _ROUTES.get(detection.doc_type)

    if extractor is None:
        return ExtractedDocument(
            source_path=str(path),
            doc_type=detection.doc_type,
            blocks=(),
            quality=QualityReport(
                doc_type=detection.doc_type,
                failures=(
                    f"unsupported_or_undetected_format: confidence={detection.confidence:.2f}, "
                    f"signals={detection.signals}",
                ),
            ),
        )

    return extractor(path, detection)
