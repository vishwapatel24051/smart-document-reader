CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    doc_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    pages_total INTEGER,
    pages_processed INTEGER NOT NULL DEFAULT 0,
    pages_needing_ocr INTEGER NOT NULL DEFAULT 0,
    pages_ocr_recovered INTEGER NOT NULL DEFAULT 0,
    tables_found INTEGER NOT NULL DEFAULT 0,
    chars_recovered INTEGER NOT NULL DEFAULT 0,
    extraction_failures TEXT[] NOT NULL DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 384 matches the default embedding model (BAAI/bge-small-en-v1.5). Changing
-- EMBEDDING_MODEL to a model with a different output dimension requires
-- changing this column (and rebuilding the index) to match.
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    section_path TEXT[] NOT NULL DEFAULT '{}',
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    from_table BOOLEAN NOT NULL DEFAULT FALSE,
    embedding VECTOR(384),
    -- Postgres's own ranking function (ts_rank_cd), not the Okapi BM25
    -- formula the rest of the project refers to as "BM25" - see README.
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
