from __future__ import annotations

from sdr.retrieval import RetrievedChunk


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def recall_at_k(retrieved: list[RetrievedChunk], expected_documents: list[str]) -> float | None:
    """1.0 if any expected document appears among the retrieved chunks'
    source documents, else 0.0. Returns None (not applicable, exclude from
    the average) for a question with no expected documents - i.e. one of
    the deliberately unanswerable questions, which recall@k can't grade.
    """
    if not expected_documents:
        return None
    retrieved_docs = {_basename(r.source_document) for r in retrieved}
    expected = {_basename(d) for d in expected_documents}
    return 1.0 if retrieved_docs & expected else 0.0


def reciprocal_rank(retrieved: list[RetrievedChunk], expected_documents: list[str]) -> float | None:
    """1/rank of the first retrieved chunk whose source document is
    expected, else 0.0. None for the same reason as recall_at_k."""
    if not expected_documents:
        return None
    expected = {_basename(d) for d in expected_documents}
    for r in retrieved:
        if _basename(r.source_document) in expected:
            return 1.0 / r.rank
    return 0.0
