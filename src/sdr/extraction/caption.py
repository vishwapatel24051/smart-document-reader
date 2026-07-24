from __future__ import annotations

import re

from .models import BlockType, ExtractedBlock

_CAPTION_RE = re.compile(r"^(table|figure|fig\.)\s*\d+", re.IGNORECASE)


def looks_like_caption(text: str) -> bool:
    return bool(_CAPTION_RE.match(text.strip()))


def pop_adjacent_caption(blocks: list[ExtractedBlock]) -> str | None:
    """Detach a trailing caption-like paragraph from ``blocks``.

    Called just before appending a table block, so a preceding "Table 1: ..."
    paragraph is attached to the table it describes instead of being kept
    (and duplicated) as an unrelated floating paragraph.
    """
    if blocks and blocks[-1].block_type == BlockType.PARAGRAPH and looks_like_caption(blocks[-1].text):
        return blocks.pop().text
    return None
