from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends

from sdr.answering import answer_query
from sdr.retrieval import search

from ..dependencies import get_db
from ..schemas import (
    AnswerResponse,
    AnswerSentenceResponse,
    CitationResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkResponse,
)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, conn: psycopg.Connection = Depends(get_db)) -> QueryResponse:
    retrieved = search(conn, req.question, k=req.k, strategy=req.strategy, rerank=req.rerank)

    answer_payload = None
    if req.with_answer:
        result = answer_query(conn, req.question, k=req.k, strategy=req.strategy, rerank=req.rerank)
        answer_payload = AnswerResponse(
            raw_text=result.raw_text,
            sentences=[
                AnswerSentenceResponse(
                    text=s.text,
                    grounded=s.grounded,
                    support_score=s.support_score,
                    citations=[
                        CitationResponse(
                            document=c.document,
                            page=c.page,
                            char_start=c.char_start,
                            char_end=c.char_end,
                            chunk_id=c.chunk_id,
                        )
                        for c in s.citations
                    ],
                )
                for s in result.sentences
            ],
        )

    return QueryResponse(
        query=req.question,
        retrieved=[
            RetrievedChunkResponse(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                source_document=r.source_document,
                text=r.text,
                page=r.page,
                section_path=list(r.section_path),
                from_table=r.from_table,
                score=r.score,
                rank=r.rank,
            )
            for r in retrieved
        ],
        answer=answer_payload,
    )
