from .models import DetectionResult, DocumentType
from .sniffer import detect_bytes, detect_file

__all__ = ["DetectionResult", "DocumentType", "detect_bytes", "detect_file"]
