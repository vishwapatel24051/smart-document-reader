from __future__ import annotations

from sdr.retrieval import RetrievedChunk

_INSTRUCTIONS = (
    "Answer the question using ONLY the numbered context passages below. "
    "Write plain prose in complete sentences. "
    "If the passages don't contain enough information to answer, say so directly. "
    "Do not use outside knowledge. Do not add a references or sources section - "
    "citations are handled separately, not by you."
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        location = chunk.source_document
        if chunk.page is not None:
            location += f", page {chunk.page}"
        blocks.append(f"[{i}] ({location})\n{chunk.text}")

    context = "\n\n".join(blocks)
    return f"{_INSTRUCTIONS}\n\nContext passages:\n{context}\n\nQuestion: {query}\n\nAnswer:"
