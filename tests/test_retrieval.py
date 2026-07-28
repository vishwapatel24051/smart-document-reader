from __future__ import annotations

import pytest

from helpers import requires_slow_tests
from sdr.ingest import ingest_document
from sdr.retrieval import rerank, search, search_dense, search_hybrid, search_lexical
from sdr.storage import repository

# The whole module needs the real embedding model (ingestion always embeds,
# regardless of which retriever is under test), so gate it once here rather
# than per-test.
pytestmark = requires_slow_tests

_CORPUS = {
    "cats.txt": (
        "Domestic cats are small carnivorous mammals often kept as household "
        "pets. They are skilled at hunting rodents."
    ),
    "finance.txt": (
        "Quarterly revenue increased due to higher enterprise software "
        "subscriptions this fiscal year."
    ),
    "weather.txt": (
        "Heavy rainfall is expected across the coastal region this weekend, "
        "with strong winds."
    ),
}


@pytest.fixture
def corpus(pg_conn, tmp_path) -> dict[str, int]:
    """Ingests a small, deliberately distinct 3-document corpus so
    retrieval results are unambiguous, and cleans it up afterward."""
    doc_ids: dict[str, int] = {}
    for name, text in _CORPUS.items():
        path = tmp_path / name
        path.write_text(text)
        outcome = ingest_document(path, pg_conn)
        assert outcome.status == "indexed"
        assert outcome.document_id is not None
        doc_ids[name] = outcome.document_id
    yield doc_ids
    for name in _CORPUS:
        repository.delete_document(pg_conn, str(tmp_path / name))


def test_lexical_search_finds_exact_keyword_match(pg_conn, corpus) -> None:
    results = search_lexical(pg_conn, "enterprise software subscriptions", k=3)
    assert results
    assert results[0].document_id == corpus["finance.txt"]


def test_dense_search_finds_semantic_paraphrase_with_no_literal_overlap(pg_conn, corpus) -> None:
    # No word here appears in cats.txt ("carnivorous", "mammals", "hunting",
    # "rodents") - this only works if search is genuinely semantic.
    results = search_dense(pg_conn, "a small pet that catches mice", k=3)
    assert results
    assert results[0].document_id == corpus["cats.txt"]


def test_lexical_search_misses_the_same_paraphrase(pg_conn, corpus) -> None:
    # The complementary failure: zero literal term overlap means Postgres's
    # lexical search has nothing to match, demonstrating why hybrid exists.
    results = search_lexical(pg_conn, "a small pet that catches mice", k=3)
    assert results == []


def test_hybrid_search_surfaces_the_dense_only_match(pg_conn, corpus) -> None:
    results = search_hybrid(pg_conn, "a small pet that catches mice", k=3)
    assert results
    assert results[0].document_id == corpus["cats.txt"]


def test_search_router_dispatches_to_the_requested_strategy(pg_conn, corpus) -> None:
    query = "a small pet that catches mice"
    direct = search_dense(pg_conn, query, k=3)
    via_router = search(pg_conn, query, k=3, strategy="dense")
    assert [r.chunk_id for r in direct] == [r.chunk_id for r in via_router]


def test_search_rejects_unknown_strategy(pg_conn, corpus) -> None:
    with pytest.raises(ValueError, match="unknown retrieval strategy"):
        search(pg_conn, "anything", strategy="not_a_real_strategy")


def test_rerank_returns_requested_count_sorted_by_new_score(pg_conn, corpus) -> None:
    candidates = search_hybrid(pg_conn, "pets and animals", k=3)
    reranked = rerank("pets and animals", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].rank == 1
    assert reranked[0].score >= reranked[1].score


def test_search_with_rerank_flag_runs_the_reranking_stage(pg_conn, corpus) -> None:
    results = search(pg_conn, "pets and animals", k=2, strategy="hybrid", rerank=True)
    assert len(results) <= 2
    assert all(r.rank == i for i, r in enumerate(results, start=1))
