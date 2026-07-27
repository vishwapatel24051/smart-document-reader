from __future__ import annotations

from dataclasses import dataclass

from sdr.extraction import ExtractedBlock, ExtractedDocument

DEFAULT_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class BlockSpan:
    block: ExtractedBlock
    start: int
    end: int


def flatten_document(
    document: ExtractedDocument, separator: str = DEFAULT_SEPARATOR
) -> tuple[str, list[BlockSpan]]:
    """Concatenate a document's blocks into one canonical text, tracking each
    block's [start, end) offset within it.

    Both chunkers key their char_start/char_end off this same flattening
    (with the same separator), so spans from the naive and structure-aware
    chunkers are directly comparable.
    """
    parts: list[str] = []
    spans: list[BlockSpan] = []
    cursor = 0
    last_index = len(document.blocks) - 1

    for i, block in enumerate(document.blocks):
        start = cursor
        parts.append(block.text)
        cursor += len(block.text)
        spans.append(BlockSpan(block=block, start=start, end=cursor))
        if i != last_index:
            parts.append(separator)
            cursor += len(separator)

    return "".join(parts), spans
