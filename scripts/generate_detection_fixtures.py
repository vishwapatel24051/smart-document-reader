"""Generates the committed binary fixtures used by tests/test_detection.py.

Run with: python scripts/generate_detection_fixtures.py
Outputs are committed to the repo; re-run only if the fixture set needs to
change.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "detection"

_ONE_PIXEL_WHITE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8ffff3f0005fe02fe0def46b80000000049454e44ae426082"
)


def _write(name: str, data: bytes) -> None:
    path = FIXTURES_DIR / name
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")


def make_pdf_with_text() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a real text layer used for detection tests.")
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf() -> bytes:
    """A page containing only a rasterized image and no text layer."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=_ONE_PIXEL_WHITE_PNG)
    data = doc.tobytes()
    doc.close()
    return data


def make_minimal_docx() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Hello from a minimal docx fixture.</w:t></w:r></w:p></w:body>"
            "</w:document>",
        )
    return buf.getvalue()


def make_non_docx_zip() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", "just a plain zip, not a docx")
    return buf.getvalue()


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    pdf_text = make_pdf_with_text()
    _write("sample_text_layer.pdf", pdf_text)
    _write("mislabeled_pdf.txt", pdf_text)  # real PDF bytes, .txt extension

    _write("sample_scanned.pdf", make_scanned_pdf())

    _write("sample.docx", make_minimal_docx())
    _write("fake.docx", make_non_docx_zip())  # valid zip, not a docx, .docx extension

    html = (
        b"<!DOCTYPE html>\n<html><head><title>Fixture</title></head>"
        b"<body><p>Hello from an html fixture.</p></body></html>"
    )
    _write("sample.html", html)

    text = b"Just a plain text fixture with a few ordinary sentences.\nSecond line.\n"
    _write("sample.txt", text)
    _write("mislabeled_txt.pdf", text)  # real text bytes, .pdf extension

    corrupt_pdf = b"%PDF-1.7\n" + (b"\x00\x01garbage-not-a-real-pdf-body" * 5)
    _write("corrupt.pdf", corrupt_pdf)

    _write("empty.txt", b"")


if __name__ == "__main__":
    main()
