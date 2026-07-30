-- Deliberately separate from the production documents/chunks tables
-- (sdr/storage/schema.sql): the harness re-indexes the same small corpus
-- under multiple (extraction, chunking) configurations, which would
-- collide with the production schema's one-row-per-source_path model.
CREATE TABLE IF NOT EXISTS eval_chunks (
    id BIGSERIAL PRIMARY KEY,
    config_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    section_path TEXT[] NOT NULL DEFAULT '{}',
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    from_table BOOLEAN NOT NULL DEFAULT FALSE,
    embedding VECTOR(384),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS eval_chunks_config_idx ON eval_chunks (config_id);
CREATE INDEX IF NOT EXISTS eval_chunks_tsv_idx ON eval_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS eval_chunks_embedding_idx ON eval_chunks USING hnsw (embedding vector_cosine_ops);
