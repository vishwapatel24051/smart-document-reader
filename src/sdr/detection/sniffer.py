from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF

from .models import DetectionResult, DocumentType

_HEADER_SCAN_BYTES = 4096
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_MIN_TEXT_CHARS_PER_PAGE = 20
_PDF_PAGES_TO_SAMPLE = 5

_HTML_STRONG_SIGNALS = ("<!doctype html", "<html")
_HTML_WEAK_SIGNALS = ("<head", "<body", "<title", "<div", "<meta", "<script", "<p>", "<a href")


def detect_file(path: Path) -> DetectionResult:
    """Classify a document's format from its content. Never raises."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return DetectionResult(DocumentType.UNKNOWN, 0.0, (f"read_error: {exc}",))
    return detect_bytes(data)


def detect_bytes(data: bytes) -> DetectionResult:
    """Classify a document's format from raw bytes. Never raises."""
    if not data:
        return DetectionResult(DocumentType.UNKNOWN, 0.0, ("empty_file",))

    header = data[:_HEADER_SCAN_BYTES]

    if _PDF_MAGIC in header[:1024]:
        return _classify_pdf(data)

    if header[:4] in _ZIP_MAGICS:
        return _classify_zip(data)

    html_result = _classify_html(header)
    if html_result is not None:
        return html_result

    return _classify_text_or_unknown(data)


def _classify_pdf(data: bytes) -> DetectionResult:
    signals = ["pdf_header_present"]
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - malformed PDFs must not crash detection
        signals.append(f"fitz_open_failed: {exc}")
        return DetectionResult(DocumentType.UNKNOWN, 0.3, tuple(signals))

    try:
        page_count = doc.page_count
        if page_count == 0:
            signals.append("pdf_has_zero_pages")
            return DetectionResult(DocumentType.UNKNOWN, 0.3, tuple(signals))

        sample_size = min(_PDF_PAGES_TO_SAMPLE, page_count)
        pages_with_text = 0
        for i in range(sample_size):
            text = doc[i].get_text().strip()
            if len(text) >= _MIN_TEXT_CHARS_PER_PAGE:
                pages_with_text += 1

        signals.append(f"sampled_{sample_size}_of_{page_count}_pages")
        signals.append(f"pages_with_text={pages_with_text}")

        if pages_with_text > 0:
            confidence = 0.7 + 0.3 * (pages_with_text / sample_size)
            return DetectionResult(DocumentType.PDF_TEXT, min(confidence, 0.99), tuple(signals))

        return DetectionResult(DocumentType.PDF_SCANNED, 0.85, tuple(signals))
    except Exception as exc:  # noqa: BLE001 - malformed page content must not crash detection
        signals.append(f"fitz_read_failed: {exc}")
        return DetectionResult(DocumentType.UNKNOWN, 0.3, tuple(signals))
    finally:
        doc.close()


def _classify_zip(data: bytes) -> DetectionResult:
    signals = ["zip_header_present"]
    try:
        names = zipfile.ZipFile(BytesIO(data)).namelist()
    except zipfile.BadZipFile as exc:
        signals.append(f"bad_zip: {exc}")
        return DetectionResult(DocumentType.UNKNOWN, 0.2, tuple(signals))

    if "word/document.xml" in names:
        signals.append("found_word/document.xml")
        return DetectionResult(DocumentType.DOCX, 0.95, tuple(signals))

    signals.append(f"zip_but_not_docx: {names[:5]}")
    return DetectionResult(DocumentType.UNKNOWN, 0.4, tuple(signals))


def _classify_html(header: bytes) -> DetectionResult | None:
    text = _best_effort_decode(header).lower().lstrip("\ufeff \t\r\n")

    strong_hits = [s for s in _HTML_STRONG_SIGNALS if s in text[:512]]
    if not strong_hits:
        return None

    weak_hits = [s for s in _HTML_WEAK_SIGNALS if s in text]
    confidence = min(0.75 + 0.05 * min(len(weak_hits), 5), 0.99)
    signals = (f"matched:{','.join(strong_hits)}", f"weak_signals={len(weak_hits)}")
    return DetectionResult(DocumentType.HTML, confidence, signals)


def _classify_text_or_unknown(data: bytes) -> DetectionResult:
    sample = data[:_HEADER_SCAN_BYTES]

    if b"\x00" in sample:
        return DetectionResult(DocumentType.UNKNOWN, 0.2, ("null_byte_in_header",))

    text = _best_effort_decode(sample)
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    ratio = printable / len(text) if text else 0.0

    if ratio >= 0.95:
        return DetectionResult(DocumentType.PLAIN_TEXT, min(0.6 + 0.3 * ratio, 0.99), (f"printable_ratio={ratio:.2f}",))

    return DetectionResult(DocumentType.UNKNOWN, 0.2, (f"printable_ratio={ratio:.2f}",))


def _best_effort_decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")
