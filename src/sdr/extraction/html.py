from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from sdr.detection import DetectionResult

from .caption import pop_adjacent_caption
from .models import BlockType, ExtractedBlock, ExtractedDocument, QualityReport, failed_extraction

_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def extract(path: Path, detection: DetectionResult) -> ExtractedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return failed_extraction(path, detection.doc_type, f"read_error: {exc}")

    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not crash extraction
        return failed_extraction(path, detection.doc_type, f"parse_error: {exc}")

    blocks: list[ExtractedBlock] = []
    failures: list[str] = []
    tables_found = 0
    heading_stack: list[tuple[int, str]] = []
    order = 0

    for tag in soup.find_all(list(_HEADING_LEVELS) + ["p", "table"]):
        if not isinstance(tag, Tag):
            continue

        if tag.name in _HEADING_LEVELS:
            text = tag.get_text(strip=True)
            if not text:
                continue
            level = _HEADING_LEVELS[tag.name]
            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, text))
            blocks.append(
                ExtractedBlock(
                    block_type=BlockType.HEADING,
                    text=text,
                    order=order,
                    section_path=tuple(h[1] for h in heading_stack),
                )
            )
            order += 1
            continue

        section_path = tuple(h[1] for h in heading_stack)

        if tag.name == "p":
            text = tag.get_text(strip=True)
            if not text:
                continue
            blocks.append(
                ExtractedBlock(block_type=BlockType.PARAGRAPH, text=text, order=order, section_path=section_path)
            )
            order += 1
            continue

        rows = _extract_table_rows(tag)
        if not rows:
            failures.append(f"empty_table_at_order_{order}")
            continue

        caption_tag = tag.find("caption")
        caption = caption_tag.get_text(strip=True) if caption_tag else pop_adjacent_caption(blocks)

        blocks.append(
            ExtractedBlock(
                block_type=BlockType.TABLE,
                text=_render_table_text(rows),
                order=order,
                section_path=section_path,
                table_rows=rows,
                caption=caption or None,
            )
        )
        tables_found += 1
        order += 1

    if not blocks:
        failures.append("no_content_recovered")

    chars_recovered = sum(len(b.text) for b in blocks)

    return ExtractedDocument(
        source_path=str(path),
        doc_type=detection.doc_type,
        blocks=tuple(blocks),
        quality=QualityReport(
            doc_type=detection.doc_type,
            pages_processed=1,
            tables_found=tables_found,
            chars_recovered=chars_recovered,
            failures=tuple(failures),
        ),
    )


def _extract_table_rows(table: Tag) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for tr in table.find_all("tr"):
        cells = tuple(cell.get_text(strip=True) for cell in tr.find_all(["td", "th"]))
        if cells:
            rows.append(cells)
    return tuple(rows)


def _render_table_text(rows: tuple[tuple[str, ...], ...]) -> str:
    return "\n".join(" | ".join(row) for row in rows)
