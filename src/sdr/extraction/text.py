from __future__ import annotations

from pathlib import Path

from sdr.detection import DetectionResult

from .models import BlockType, ExtractedBlock, ExtractedDocument, QualityReport, failed_extraction


def extract(path: Path, detection: DetectionResult) -> ExtractedDocument:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return failed_extraction(path, detection.doc_type, f"read_error: {exc}")

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    blocks = tuple(
        ExtractedBlock(block_type=BlockType.PARAGRAPH, text=p, order=i)
        for i, p in enumerate(paragraphs)
    )

    failures = () if blocks else ("no_paragraphs_recovered",)
    chars_recovered = sum(len(b.text) for b in blocks)

    return ExtractedDocument(
        source_path=str(path),
        doc_type=detection.doc_type,
        blocks=blocks,
        quality=QualityReport(
            doc_type=detection.doc_type,
            pages_processed=1,
            chars_recovered=chars_recovered,
            failures=failures,
        ),
    )
