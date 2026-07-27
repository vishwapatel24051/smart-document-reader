from __future__ import annotations

from pathlib import Path

from sdr.chunking import chunk_naive, chunk_structure_aware, flatten_document
from sdr.extraction import extract

EXTRACTION_FIXTURES = Path(__file__).parent / "fixtures" / "extraction"
DETECTION_FIXTURES = Path(__file__).parent / "fixtures" / "detection"


def test_structure_aware_never_splits_a_table() -> None:
    document = extract(EXTRACTION_FIXTURES / "table.pdf")
    chunks = chunk_structure_aware(document)

    table_chunks = [c for c in chunks if c.from_table]
    assert len(table_chunks) == 1
    assert table_chunks[0].text == "Trial | Outcome\n1 | alpha\n2 | beta"

    # The table's content must not leak into any non-table chunk either.
    for chunk in chunks:
        if not chunk.from_table:
            assert "alpha" not in chunk.text
            assert "beta" not in chunk.text


def test_structure_aware_drops_orphan_heading_before_a_table() -> None:
    # In structured.docx/.html, "Results" is immediately followed by the
    # table (its caption gets absorbed at extraction time) - the heading
    # must not surface as a standalone, content-free chunk.
    document = extract(EXTRACTION_FIXTURES / "structured.docx")
    chunks = chunk_structure_aware(document)
    assert not any(c.text.strip() == "Results" for c in chunks)
    assert any(c.from_table and c.section_path == ("Introduction", "Results") for c in chunks)


def test_structure_aware_char_spans_match_flattened_text() -> None:
    for filename in ["structured.docx", "structured.html", "table.pdf", "multi_column.pdf"]:
        document = extract(EXTRACTION_FIXTURES / filename)
        full_text, _ = flatten_document(document)
        for chunk in chunk_structure_aware(document):
            assert full_text[chunk.char_start : chunk.char_end] == chunk.text


def test_structure_aware_chunk_metadata_matches_source_blocks() -> None:
    document = extract(EXTRACTION_FIXTURES / "structured.html")
    chunks = chunk_structure_aware(document)

    first = chunks[0]
    assert first.source_document == str(EXTRACTION_FIXTURES / "structured.html")
    assert first.section_path == ("Introduction",)
    assert first.strategy == "structure_aware"
    assert first.chunk_index == 0


def test_structure_aware_respects_multi_column_reading_order() -> None:
    document = extract(EXTRACTION_FIXTURES / "multi_column.pdf")
    chunks = chunk_structure_aware(document, max_chars=10_000)
    assert len(chunks) == 1
    assert chunks[0].text.index("Left column, item three.") < chunks[0].text.index("Right column, item one.")


def test_naive_chunker_can_split_a_table_that_structure_aware_keeps_whole() -> None:
    document = extract(EXTRACTION_FIXTURES / "table.pdf")
    table_text = next(b.text for b in document.blocks if b.block_type.value == "table")

    naive_chunks = chunk_naive(document, chunk_size=15, overlap=0)
    aware_chunks = chunk_structure_aware(document)

    assert not any(table_text in c.text for c in naive_chunks)
    assert any(table_text == c.text for c in aware_chunks)


def test_naive_chunker_spans_slice_the_flattened_text_directly() -> None:
    document = extract(EXTRACTION_FIXTURES / "structured.docx")
    full_text, _ = flatten_document(document)
    for chunk in chunk_naive(document, chunk_size=40, overlap=5):
        assert full_text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.strategy == "naive_fixed_size"


def test_naive_chunker_windows_cover_the_full_text_with_overlap() -> None:
    document = extract(EXTRACTION_FIXTURES / "structured.html")
    chunks = chunk_naive(document, chunk_size=40, overlap=10)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(flatten_document(document)[0])
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.char_start <= prev.char_end  # consecutive windows overlap, never gap


def test_both_chunkers_return_nothing_for_a_document_with_no_blocks() -> None:
    document = extract(DETECTION_FIXTURES / "empty.txt")
    assert document.blocks == ()
    assert chunk_structure_aware(document) == []
    assert chunk_naive(document) == []
