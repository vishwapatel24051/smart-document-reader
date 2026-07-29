from __future__ import annotations

import psycopg

from sdr.retrieval import search as retrieve

from . import llm
from .groundedness import ground_sentence, split_sentences
from .models import Answer
from .prompt import build_prompt


def answer_query(
    conn: psycopg.Connection,
    query: str,
    k: int = 5,
    strategy: str | None = None,
    rerank: bool | None = None,
    groundedness_threshold: float = 0.4,
) -> Answer:
    """Retrieve, generate, and ground: the full answering pipeline.

    Returns an Answer with no sentences (not an exception) if nothing was
    retrieved. LLM/network failures (e.g. Ollama not running) propagate as
    httpx.HTTPError - this function doesn't swallow them.
    """
    chunks = retrieve(conn, query, k=k, strategy=strategy, rerank=rerank)
    if not chunks:
        return Answer(query=query, sentences=(), raw_text="", retrieved_chunk_ids=())

    raw_text = llm.generate(build_prompt(query, chunks))
    sentences = tuple(
        ground_sentence(s, chunks, threshold=groundedness_threshold) for s in split_sentences(raw_text)
    )

    return Answer(
        query=query,
        sentences=sentences,
        raw_text=raw_text,
        retrieved_chunk_ids=tuple(c.chunk_id for c in chunks),
    )
