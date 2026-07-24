from __future__ import annotations

import shutil
from io import BytesIO

_tesseract_available: bool | None = None


def tesseract_available() -> bool:
    """Whether a tesseract binary is on PATH. Cached after the first check."""
    global _tesseract_available
    if _tesseract_available is None:
        _tesseract_available = shutil.which("tesseract") is not None
    return _tesseract_available


def run_ocr(png_bytes: bytes) -> str:
    """Run OCR on a rendered page image.

    Callers should check tesseract_available() first; this raises whatever
    pytesseract/PIL raise (e.g. if tesseract isn't installed) rather than
    swallowing errors, so callers can record the real failure reason.
    """
    import pytesseract
    from PIL import Image

    image = Image.open(BytesIO(png_bytes))
    return pytesseract.image_to_string(image).strip()
