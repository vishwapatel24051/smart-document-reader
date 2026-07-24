from __future__ import annotations

from pathlib import Path

import pytest

from sdr.detection import DocumentType, detect_bytes, detect_file

FIXTURES = Path(__file__).parent / "fixtures" / "detection"


@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("sample_text_layer.pdf", DocumentType.PDF_TEXT),
        ("sample_scanned.pdf", DocumentType.PDF_SCANNED),
        ("sample.docx", DocumentType.DOCX),
        ("sample.html", DocumentType.HTML),
        ("sample.txt", DocumentType.PLAIN_TEXT),
    ],
)
def test_detects_well_formed_files(filename: str, expected_type: DocumentType) -> None:
    result = detect_file(FIXTURES / filename)
    assert result.doc_type == expected_type
    assert result.confidence > 0.5


def test_mislabeled_pdf_saved_with_txt_extension_is_detected_by_content() -> None:
    # Real PDF bytes, .txt extension - detection must ignore the extension.
    result = detect_file(FIXTURES / "mislabeled_pdf.txt")
    assert result.doc_type == DocumentType.PDF_TEXT


def test_mislabeled_text_saved_with_pdf_extension_is_detected_by_content() -> None:
    # Real plain-text bytes, .pdf extension - detection must ignore the extension.
    result = detect_file(FIXTURES / "mislabeled_txt.pdf")
    assert result.doc_type == DocumentType.PLAIN_TEXT


def test_zip_that_is_not_docx_is_not_classified_as_docx() -> None:
    result = detect_file(FIXTURES / "fake.docx")
    assert result.doc_type != DocumentType.DOCX


def test_corrupt_pdf_does_not_crash_and_reports_low_confidence() -> None:
    result = detect_file(FIXTURES / "corrupt.pdf")
    assert result.doc_type == DocumentType.UNKNOWN
    assert result.confidence < 0.5
    assert any("pdf" in signal for signal in result.signals)


def test_empty_file_does_not_crash() -> None:
    result = detect_file(FIXTURES / "empty.txt")
    assert result.doc_type == DocumentType.UNKNOWN
    assert result.confidence == 0.0


def test_nonexistent_file_does_not_crash() -> None:
    result = detect_file(FIXTURES / "does_not_exist.pdf")
    assert result.doc_type == DocumentType.UNKNOWN
    assert result.confidence == 0.0


def test_detect_bytes_matches_detect_file_for_html() -> None:
    data = (FIXTURES / "sample.html").read_bytes()
    result = detect_bytes(data)
    assert result.doc_type == DocumentType.HTML
