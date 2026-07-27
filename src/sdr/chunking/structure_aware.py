from __future__ import annotations

from sdr.extraction import BlockType, ExtractedBlock, ExtractedDocument

from .flatten import DEFAULT_SEPARATOR, flatten_document
from .models import Chunk

# Soft target, not a hard cutoff: a single paragraph or table larger than
# this still becomes its own chunk rather than being split mid-unit. Not
# tuned against any measurement yet - Phase 7 is where chunk size actually
# gets evaluated.
_DEFAULT_MAX_CHARS = 1200


def chunk_structure_aware(
    document: ExtractedDocument, max_chars: int = _DEFAULT_MAX_CHARS, separator: str = DEFAULT_SEPARATOR
) -> list[Chunk]:
    """Chunk on semantic boundaries: a table is always its own chunk, a
    heading always starts a new chunk, and paragraphs accumulate into the
    current chunk until adding the next one would exceed max_chars.

    Never splits a table or a single paragraph mid-unit - max_chars is a
    soft target, so a chunk can exceed it when one semantic unit is bigger
    than the budget.
    """
    _, spans = flatten_document(document, separator=separator)
    span_by_block = {bs.block: bs for bs in spans}

    chunks: list[Chunk] = []
    current: list[ExtractedBlock] = []
    chunk_index = 0

    def has_content(blocks: list[ExtractedBlock]) -> bool:
        return any(b.block_type != BlockType.HEADING for b in blocks)

    def current_chars(blocks: list[ExtractedBlock]) -> int:
        return sum(len(b.text) for b in blocks)

    def flush() -> None:
        nonlocal current, chunk_index
        if not current:
            return
        if not has_content(current):
            # A heading with nothing accumulated after it (e.g. immediately
            # followed by a table) is dropped rather than emitted as a
            # content-free chunk; its section_path already reaches
            # downstream blocks via their own extraction-time metadata.
            current = []
            return
        first_span = span_by_block[current[0]]
        last_span = span_by_block[current[-1]]
        chunks.append(
            Chunk(
                text=separator.join(b.text for b in current),
                source_document=document.source_path,
                page=current[0].page,
                section_path=current[0].section_path,
                char_start=first_span.start,
                char_end=last_span.end,
                from_table=False,
                chunk_index=chunk_index,
                strategy="structure_aware",
            )
        )
        chunk_index += 1
        current = []

    for block in document.blocks:
        if block.block_type == BlockType.TABLE:
            flush()
            span = span_by_block[block]
            chunks.append(
                Chunk(
                    text=block.text,
                    source_document=document.source_path,
                    page=block.page,
                    section_path=block.section_path,
                    char_start=span.start,
                    char_end=span.end,
                    from_table=True,
                    chunk_index=chunk_index,
                    strategy="structure_aware",
                )
            )
            chunk_index += 1
            continue

        if block.block_type == BlockType.HEADING:
            flush()
            current.append(block)
            continue

        # PARAGRAPH: flush on overflow, but only once the current chunk
        # already has real content - a heading is never left stranded alone
        # just because the next paragraph is large.
        projected = current_chars(current) + len(block.text)
        if current and has_content(current) and projected > max_chars:
            flush()
        current.append(block)

    flush()
    return chunks
