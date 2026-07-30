from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    source_path: str
    status: str
    document_id: int | None
    chunks_indexed: int
    reason: str | None = None


class DocumentSummary(BaseModel):
    id: int
    source_path: str
    doc_type: str
    content_hash: str
    pages_total: int | None
    pages_processed: int
    pages_needing_ocr: int
    pages_ocr_recovered: int
    tables_found: int
    chars_recovered: int
    extraction_failures: list[str]
    ingested_at: datetime
    updated_at: datetime


class RetrievedChunkResponse(BaseModel):
    chunk_id: int
    document_id: int
    source_document: str
    text: str
    page: int | None
    section_path: list[str]
    from_table: bool
    score: float
    rank: int


class CitationResponse(BaseModel):
    document: str
    page: int | None
    char_start: int
    char_end: int
    chunk_id: int


class AnswerSentenceResponse(BaseModel):
    text: str
    grounded: bool
    support_score: float
    citations: list[CitationResponse]


class AnswerResponse(BaseModel):
    raw_text: str
    sentences: list[AnswerSentenceResponse]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    strategy: str | None = Field(default=None, description='"dense", "lexical", or "hybrid"; defaults to config')
    rerank: bool | None = Field(default=None, description="defaults to config")
    with_answer: bool = Field(default=True, description="also generate an LLM answer with citations")


class QueryResponse(BaseModel):
    query: str
    retrieved: list[RetrievedChunkResponse]
    answer: AnswerResponse | None = None


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    pgvector: bool
    ollama: bool
