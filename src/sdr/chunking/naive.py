from __future__ import annotations

from sdr.extraction import BlockType, ExtractedDocument

from .flatten import BlockSpan, DEFAULT_SEPARATOR, flatten_document
from .models import Chunk

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_OVERLAP = 100


def chunk_naive(
    document: ExtractedDocument,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
    separator: str = DEFAULT_SEPARATOR,
) -> list[Chunk]:
    """A deliberately naive fixed-size chunker: slides a fixed character
    window over the flattened document text with no awareness of
    paragraphs, tables, or section boundaries.

    This exists as the baseline Phase 7 measures the structure-aware
    chunker against - it is not a recommended strategy. It will split
    mid-sentence and mid-table.
    """
    full_text, spans = flatten_document(document, separator=separator)
    if not full_text.strip():
        return []

    step = max(chunk_size - overlap, 1)
    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0
    n = len(full_text)

    while start < n:
        end = min(start + chunk_size, n)
        text = full_text[start:end]
        if text.strip():
            page, section_path, from_table = _metadata_for_span(spans, start, end)
            chunks.append(
                Chunk(
                    text=text,
                    source_document=document.source_path,
                    page=page,
                    section_path=section_path,
                    char_start=start,
                    char_end=end,
                    from_table=from_table,
                    chunk_index=chunk_index,
                    strategy="naive_fixed_size",
                )
            )
            chunk_index += 1
        if end == n:
            break
        start += step

    return chunks


def _metadata_for_span(
    spans: list[BlockSpan], start: int, end: int
) -> tuple[int | None, tuple[str, ...], bool]:
    """Best-effort metadata: attribute the window to whichever block it
    overlaps the most. This is approximate by construction - a naive window
    can straddle a section or table boundary, which is exactly the failure
    mode this baseline exists to demonstrate.
    """
    best: BlockSpan | None = None
    best_overlap = -1
    for span in spans:
        overlap = min(end, span.end) - max(start, span.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = span

    if best is None:
        return None, (), False
    return best.block.page, best.block.section_path, best.block.block_type == BlockType.TABLE
