from __future__ import annotations

from pathlib import Path

from sdr.extraction import BlockType, extract
from sdr.extraction.ocr import tesseract_available

EXTRACTION_FIXTURES = Path(__file__).parent / "fixtures" / "extraction"
DETECTION_FIXTURES = Path(__file__).parent / "fixtures" / "detection"


def test_multi_column_pdf_preserves_reading_order() -> None:
    result = extract(EXTRACTION_FIXTURES / "multi_column.pdf")
    texts = [b.text for b in result.blocks]

    assert texts == [
        "Left column, item one.",
        "Left column, item two.",
        "Left column, item three.",
        "Right column, item one.",
        "Right column, item two.",
        "Right column, item three.",
    ]
    assert result.quality.failures == ()


def test_pdf_table_is_structured_not_flattened_and_keeps_its_caption() -> None:
    result = extract(EXTRACTION_FIXTURES / "table.pdf")

    tables = [b for b in result.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0]
    assert table.table_rows == (
        ("Trial", "Outcome"),
        ("1", "alpha"),
        ("2", "beta"),
    )
    assert table.caption == "Table 1: Example results"

    # The caption must not also survive as an unrelated floating paragraph.
    paragraph_texts = [b.text for b in result.blocks if b.block_type == BlockType.PARAGRAPH]
    assert "Table 1: Example results" not in paragraph_texts

    headings = [b for b in result.blocks if b.block_type == BlockType.HEADING]
    assert [h.text for h in headings] == ["Experiment Results"]
    assert table.section_path == ("Experiment Results",)
    assert result.quality.tables_found == 1


def test_pdf_table_bbox_text_is_not_duplicated_as_a_paragraph() -> None:
    result = extract(EXTRACTION_FIXTURES / "table.pdf")
    paragraph_texts = " ".join(b.text for b in result.blocks if b.block_type == BlockType.PARAGRAPH)
    assert "alpha" not in paragraph_texts
    assert "beta" not in paragraph_texts


def test_scanned_pdf_triggers_ocr_and_reports_outcome_honestly() -> None:
    result = extract(EXTRACTION_FIXTURES / "scanned_text.pdf")
    assert result.quality.pages_needing_ocr == 1

    if tesseract_available():
        assert result.quality.pages_ocr_recovered == 1
        ocr_blocks = [b for b in result.blocks if b.from_ocr]
        assert len(ocr_blocks) == 1
        assert ocr_blocks[0].text  # some text was recovered; exact OCR output isn't asserted
    else:
        assert result.quality.pages_ocr_recovered == 0
        assert any("tesseract_not_installed" in f for f in result.quality.failures)


def test_docx_preserves_heading_hierarchy_table_and_caption() -> None:
    result = extract(EXTRACTION_FIXTURES / "structured.docx")

    headings = [b for b in result.blocks if b.block_type == BlockType.HEADING]
    assert [h.text for h in headings] == ["Introduction", "Results"]

    tables = [b for b in result.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    assert tables[0].caption == "Table 1: Summary of results"
    assert tables[0].table_rows == (
        ("Trial", "Outcome"),
        ("1", "alpha"),
        ("2", "beta"),
    )
    assert tables[0].section_path == ("Introduction", "Results")

    # Reading order: heading, paragraph, heading, table, paragraph.
    assert [b.block_type for b in result.blocks] == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.HEADING,
        BlockType.TABLE,
        BlockType.PARAGRAPH,
    ]


def test_html_preserves_heading_hierarchy_table_and_native_caption() -> None:
    result = extract(EXTRACTION_FIXTURES / "structured.html")

    headings = [b for b in result.blocks if b.block_type == BlockType.HEADING]
    assert [h.text for h in headings] == ["Introduction", "Results"]

    tables = [b for b in result.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    assert tables[0].caption == "Table 1: Summary of results"
    assert tables[0].table_rows == (
        ("Trial", "Outcome"),
        ("1", "alpha"),
        ("2", "beta"),
    )
    assert tables[0].section_path == ("Introduction", "Results")


def test_plain_text_splits_on_blank_lines() -> None:
    result = extract(EXTRACTION_FIXTURES / "structured.txt")
    texts = [b.text for b in result.blocks]
    assert texts == [
        "This document exercises the plain text extractor.",
        "It has a few short paragraphs separated by blank lines.",
        "This is the third and final paragraph.",
    ]
    assert all(b.block_type == BlockType.PARAGRAPH for b in result.blocks)


def test_corrupt_pdf_does_not_crash_and_surfaces_failure() -> None:
    result = extract(DETECTION_FIXTURES / "corrupt.pdf")
    assert result.blocks == ()
    assert result.quality.failures != ()


def test_empty_file_does_not_crash_and_surfaces_failure() -> None:
    result = extract(DETECTION_FIXTURES / "empty.txt")
    assert result.blocks == ()
    assert result.quality.failures != ()


def test_router_dispatches_by_content_not_by_extension() -> None:
    # Real PDF bytes saved with a .txt extension - the router must not be
    # fooled by the extension, matching detection's own guarantee.
    result = extract(DETECTION_FIXTURES / "mislabeled_pdf.txt")
    assert result.blocks != ()
    assert result.blocks[0].block_type == BlockType.PARAGRAPH
