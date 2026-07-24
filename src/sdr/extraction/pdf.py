from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from sdr.detection import DetectionResult

from . import ocr
from .caption import pop_adjacent_caption
from .models import BlockType, ExtractedBlock, ExtractedDocument, QualityReport, failed_extraction

# A page with fewer recovered characters than this is treated as image-only
# and sent through OCR instead.
_MIN_CHARS_TO_SKIP_OCR = 20

# A text block is treated as a heading if its dominant font size is at least
# this many times the document's median span size, and it's short.
_HEADING_SIZE_RATIO = 1.15
_HEADING_MAX_WORDS = 15

# A vertical strip is considered a column gutter if it falls within this
# fraction of the page width and no block bbox crosses it.
_GUTTER_SEARCH_MIN_X_FRAC = 0.35
_GUTTER_SEARCH_MAX_X_FRAC = 0.65

_OCR_RENDER_DPI = 200


def extract(path: Path, detection: DetectionResult) -> ExtractedDocument:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return failed_extraction(path, detection.doc_type, f"read_error: {exc}")

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - malformed PDFs must not crash extraction
        return failed_extraction(path, detection.doc_type, f"open_error: {exc}")

    try:
        return _extract_pages(doc, path, detection)
    finally:
        doc.close()


def _extract_pages(doc: fitz.Document, path: Path, detection: DetectionResult) -> ExtractedDocument:
    page_dicts = [doc[i].get_text("dict") for i in range(doc.page_count)]
    median_font_size = _median_font_size(page_dicts)

    blocks: list[ExtractedBlock] = []
    failures: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    tables_found = 0
    pages_needing_ocr = 0
    pages_ocr_recovered = 0
    order = 0

    for page_index, page_dict in enumerate(page_dicts):
        page = doc[page_index]
        page_number = page_index + 1

        try:
            tables = list(page.find_tables().tables)
        except Exception as exc:  # noqa: BLE001 - table detection must not crash extraction
            tables = []
            failures.append(f"page_{page_number}_table_detection_failed: {exc}")

        table_bboxes = [fitz.Rect(t.bbox) for t in tables]
        text_items = _collect_text_items(page_dict, table_bboxes, median_font_size)
        table_items: list[dict[str, Any]] = [{"kind": "table", "bbox": fitz.Rect(t.bbox), "table": t} for t in tables]
        items = _order_items(text_items + table_items, page.rect.width)

        chars_this_page = sum(len(it["text"]) for it in items if it["kind"] == "text")
        if chars_this_page < _MIN_CHARS_TO_SKIP_OCR:
            pages_needing_ocr += 1
            ocr_text, ocr_error = _run_page_ocr(page)
            if ocr_error:
                failures.append(f"page_{page_number}_ocr_failed: {ocr_error}")
            elif ocr_text:
                pages_ocr_recovered += 1
                blocks.append(
                    ExtractedBlock(
                        block_type=BlockType.PARAGRAPH,
                        text=ocr_text,
                        order=order,
                        section_path=tuple(h[1] for h in heading_stack),
                        page=page_number,
                        from_ocr=True,
                    )
                )
                order += 1
            else:
                failures.append(f"page_{page_number}_no_text_and_ocr_empty")

        for item in items:
            if item["kind"] == "table":
                rows = _extract_table_rows(item["table"])
                if not rows:
                    continue
                caption = pop_adjacent_caption(blocks)
                blocks.append(
                    ExtractedBlock(
                        block_type=BlockType.TABLE,
                        text="\n".join(" | ".join(r) for r in rows),
                        order=order,
                        section_path=tuple(h[1] for h in heading_stack),
                        page=page_number,
                        table_rows=rows,
                        caption=caption,
                    )
                )
                tables_found += 1
                order += 1
                continue

            text = item["text"]
            if item["is_heading"]:
                heading_stack[:] = []
                heading_stack.append((1, text))
                blocks.append(
                    ExtractedBlock(
                        block_type=BlockType.HEADING,
                        text=text,
                        order=order,
                        section_path=tuple(h[1] for h in heading_stack),
                        page=page_number,
                    )
                )
            else:
                blocks.append(
                    ExtractedBlock(
                        block_type=BlockType.PARAGRAPH,
                        text=text,
                        order=order,
                        section_path=tuple(h[1] for h in heading_stack),
                        page=page_number,
                    )
                )
            order += 1

    if not blocks:
        failures.append("no_content_recovered_from_any_page")

    chars_recovered = sum(len(b.text) for b in blocks)

    return ExtractedDocument(
        source_path=str(path),
        doc_type=detection.doc_type,
        blocks=tuple(blocks),
        quality=QualityReport(
            doc_type=detection.doc_type,
            pages_total=doc.page_count,
            pages_processed=doc.page_count,
            pages_needing_ocr=pages_needing_ocr,
            pages_ocr_recovered=pages_ocr_recovered,
            tables_found=tables_found,
            chars_recovered=chars_recovered,
            failures=tuple(failures),
        ),
    )


def _median_font_size(page_dicts: list[dict[str, Any]]) -> float:
    sizes: list[float] = []
    for page_dict in page_dicts:
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(span["size"])
    return statistics.median(sizes) if sizes else 10.0


def _collect_text_items(
    page_dict: dict[str, Any], table_bboxes: list[fitz.Rect], median_font_size: float
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        bbox = fitz.Rect(block["bbox"])
        if _overlaps_a_table(bbox, table_bboxes):
            continue  # already captured structurally via find_tables()

        lines = block.get("lines", [])
        text = " ".join(
            " ".join(span.get("text", "") for span in line.get("spans", [])) for line in lines
        ).strip()
        text = " ".join(text.split())
        if not text:
            continue

        sizes = [span["size"] for line in lines for span in line.get("spans", []) if span.get("text", "").strip()]
        block_size = max(sizes) if sizes else 0.0
        word_count = len(text.split())
        is_heading = (
            block_size >= median_font_size * _HEADING_SIZE_RATIO
            and word_count <= _HEADING_MAX_WORDS
            and not text.endswith((".", ",", ";"))
        )

        items.append({"kind": "text", "bbox": bbox, "text": text, "is_heading": is_heading})

    return items


def _overlaps_a_table(bbox: fitz.Rect, table_bboxes: list[fitz.Rect]) -> bool:
    for table_bbox in table_bboxes:
        intersection = bbox & table_bbox
        if intersection.is_empty:
            continue
        if intersection.get_area() >= 0.5 * bbox.get_area():
            return True
    return False


def _order_items(items: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    if not items:
        return items

    gutter_x = _find_gutter(items, page_width)
    if gutter_x is None:
        return sorted(items, key=lambda it: (it["bbox"].y0, it["bbox"].x0))

    def column(it: dict[str, Any]) -> int:
        center_x = (it["bbox"].x0 + it["bbox"].x1) / 2
        return 0 if center_x < gutter_x else 1

    return sorted(items, key=lambda it: (column(it), it["bbox"].y0, it["bbox"].x0))


def _find_gutter(items: list[dict[str, Any]], page_width: float) -> float | None:
    search_min = page_width * _GUTTER_SEARCH_MIN_X_FRAC
    search_max = page_width * _GUTTER_SEARCH_MAX_X_FRAC
    if search_max <= search_min:
        return None

    straddles_gutter = any(it["bbox"].x0 < search_max and it["bbox"].x1 > search_min for it in items)
    if straddles_gutter:
        return None  # something spans the middle strip - not a clean 2-column layout

    left_present = any(it["bbox"].x1 <= search_min for it in items)
    right_present = any(it["bbox"].x0 >= search_max for it in items)
    if left_present and right_present:
        return (search_min + search_max) / 2
    return None


def _extract_table_rows(table: Any) -> tuple[tuple[str, ...], ...]:
    try:
        raw_rows = table.extract()
    except Exception:  # noqa: BLE001 - a single bad table must not crash extraction
        return ()
    rows = tuple(tuple((cell or "").strip() for cell in row) for row in raw_rows)
    return tuple(r for r in rows if any(c for c in r))


def _run_page_ocr(page: fitz.Page) -> tuple[str, str | None]:
    if not ocr.tesseract_available():
        return "", "tesseract_not_installed"
    try:
        pixmap = page.get_pixmap(dpi=_OCR_RENDER_DPI)
        text = ocr.run_ocr(pixmap.tobytes("png"))
        return text, None
    except Exception as exc:  # noqa: BLE001
        return "", str(exc)
