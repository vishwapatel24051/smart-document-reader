from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psycopg

from sdr.answering import llm
from sdr.answering.groundedness import ground_sentence, split_sentences
from sdr.answering.prompt import build_prompt
from sdr.retrieval import rerank as rerank_candidates

from . import metrics as m
from .indexer import CHUNK_STRATEGIES, EXTRACTION_STRATEGIES, index_corpus, index_id
from .retrieval import search as eval_search

RETRIEVAL_STRATEGIES = ("dense", "hybrid")
RERANK_OPTIONS = (False, True)
DEFAULT_K = 5

# How much wider a candidate pool to pull before reranking narrows it back
# down to k - matches sdr.retrieval.router's _RERANK_POOL_MULTIPLIER.
_RERANK_POOL_MULTIPLIER = 3


@dataclass(frozen=True)
class QuestionRecord:
    id: str
    question: str
    expected_documents: list[str]
    category: str


def load_questions(path: Path) -> list[QuestionRecord]:
    data = json.loads(path.read_text())
    return [
        QuestionRecord(
            id=q["id"],
            question=q["question"],
            expected_documents=q["expected_documents"],
            category=q["category"],
        )
        for q in data["questions"]
    ]


@dataclass(frozen=True)
class ConfigResult:
    extraction: str
    chunking: str
    retrieval: str
    rerank: bool
    n_questions: int
    n_graded: int
    recall_at_k: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    groundedness_rate: float | None = None
    unanswerable_decline_rate: float | None = None
    avg_response_tokens: float | None = None
    total_wall_seconds: float = 0.0


def run_matrix(
    conn: psycopg.Connection,
    corpus_dir: Path,
    questions: list[QuestionRecord],
    k: int = DEFAULT_K,
    with_answers: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> list[ConfigResult]:
    """Runs every (extraction, chunking, retrieval, rerank) combination
    against the full question set. Re-indexes the corpus once per
    (extraction, chunking) pair and reuses it across the retrieval/rerank
    combinations that share it, rather than re-indexing per combination.
    """
    results: list[ConfigResult] = []

    for extraction in EXTRACTION_STRATEGIES:
        for chunking in CHUNK_STRATEGIES:
            config_id = index_id(extraction, chunking)
            if on_progress:
                on_progress(f"indexing corpus: extraction={extraction} chunking={chunking}")
            index_corpus(conn, corpus_dir, extraction, chunking)

            for retrieval in RETRIEVAL_STRATEGIES:
                for rerank in RERANK_OPTIONS:
                    if on_progress:
                        on_progress(f"  running: retrieval={retrieval} rerank={rerank}")
                    results.append(
                        _run_one_config(
                            conn, config_id, extraction, chunking, retrieval, rerank, questions, k, with_answers
                        )
                    )
    return results


def _run_one_config(
    conn: psycopg.Connection,
    config_id: str,
    extraction: str,
    chunking: str,
    retrieval: str,
    rerank: bool,
    questions: list[QuestionRecord],
    k: int,
    with_answers: bool,
) -> ConfigResult:
    recalls: list[float] = []
    rrs: list[float] = []
    latencies_ms: list[float] = []
    grounded_flags: list[bool] = []
    unanswerable_correct: list[bool] = []
    response_token_counts: list[int] = []
    wall_start = time.monotonic()

    pool_k = k * _RERANK_POOL_MULTIPLIER if rerank else k

    for q in questions:
        t0 = time.monotonic()
        candidates = eval_search(conn, config_id, q.question, pool_k, retrieval)
        if rerank:
            candidates = rerank_candidates(q.question, candidates, k)
        else:
            candidates = candidates[:k]
        latencies_ms.append((time.monotonic() - t0) * 1000)

        recall = m.recall_at_k(candidates, q.expected_documents)
        rr = m.reciprocal_rank(candidates, q.expected_documents)
        if recall is not None:
            recalls.append(recall)
            rrs.append(rr)  # type: ignore[arg-type]

        if with_answers:
            if not candidates:
                unanswerable_correct.append(len(q.expected_documents) == 0)
                continue
            gen = llm.generate_with_stats(build_prompt(q.question, candidates))
            if gen.response_tokens is not None:
                response_token_counts.append(gen.response_tokens)
            sentences = [ground_sentence(s, candidates) for s in split_sentences(gen.text)]
            any_grounded = any(s.grounded for s in sentences)
            grounded_flags.append(any_grounded)
            if not q.expected_documents:
                # Correct behavior for an unanswerable question is to NOT
                # produce a grounded, cited claim.
                unanswerable_correct.append(not any_grounded)

    return ConfigResult(
        extraction=extraction,
        chunking=chunking,
        retrieval=retrieval,
        rerank=rerank,
        n_questions=len(questions),
        n_graded=len(recalls),
        recall_at_k=statistics.mean(recalls) if recalls else 0.0,
        mrr=statistics.mean(rrs) if rrs else 0.0,
        latency_p50_ms=_percentile(latencies_ms, 50),
        latency_p95_ms=_percentile(latencies_ms, 95),
        groundedness_rate=statistics.mean(grounded_flags) if with_answers and grounded_flags else None,
        unanswerable_decline_rate=(
            statistics.mean(unanswerable_correct) if with_answers and unanswerable_correct else None
        ),
        avg_response_tokens=statistics.mean(response_token_counts) if response_token_counts else None,
        total_wall_seconds=time.monotonic() - wall_start,
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return ordered[idx]
