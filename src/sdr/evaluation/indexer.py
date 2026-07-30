from __future__ import annotations

from pathlib import Path

import psycopg

from sdr.chunking import chunk_naive, chunk_structure_aware
from sdr.embedding import embed_texts
from sdr.extraction import extract, naive_extract

_EXTRACTORS = {"naive": naive_extract, "structure_aware": extract}
_CHUNKERS = {"naive": chunk_naive, "structure_aware": chunk_structure_aware}

EXTRACTION_STRATEGIES = tuple(_EXTRACTORS)
CHUNK_STRATEGIES = tuple(_CHUNKERS)


def index_id(extraction_strategy: str, chunk_strategy: str) -> str:
    return f"{extraction_strategy}-{chunk_strategy}"


def index_corpus(
    conn: psycopg.Connection, corpus_dir: Path, extraction_strategy: str, chunk_strategy: str
) -> int:
    """(Re)builds eval_chunks for one (extraction, chunking) configuration.

    Wipes any prior rows under that config_id first, so re-running the
    harness after editing the corpus doesn't leave stale chunks behind.
    Returns the number of chunks indexed.
    """
    config_id = index_id(extraction_strategy, chunk_strategy)
    extractor = _EXTRACTORS[extraction_strategy]
    chunker = _CHUNKERS[chunk_strategy]

    conn.execute("DELETE FROM eval_chunks WHERE config_id = %s", (config_id,))

    total_chunks = 0
    with conn.cursor() as cur:
        for path in sorted(corpus_dir.iterdir()):
            if path.is_dir() or path.name.startswith("_") or path.name.startswith("."):
                continue

            document = extractor(path)
            if not document.blocks:
                continue

            chunks = chunker(document)
            if not chunks:
                continue

            embeddings = embed_texts([c.text for c in chunks])
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                cur.execute(
                    """
                    INSERT INTO eval_chunks (
                        config_id, source_path, doc_type, chunk_index, strategy, text,
                        page, section_path, char_start, char_end, from_table, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        config_id,
                        chunk.source_document,
                        document.doc_type.value,
                        chunk.chunk_index,
                        chunk.strategy,
                        chunk.text,
                        chunk.page,
                        list(chunk.section_path),
                        chunk.char_start,
                        chunk.char_end,
                        chunk.from_table,
                        embedding,
                    ),
                )
                total_chunks += 1

    return total_chunks
