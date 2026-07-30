from __future__ import annotations

from pathlib import Path

from helpers import requires_slow_tests
from sdr.evaluation import ensure_eval_schema, index_corpus, load_questions
from sdr.evaluation.metrics import reciprocal_rank, recall_at_k
from sdr.evaluation.retrieval import search as eval_search
from sdr.retrieval import RetrievedChunk

EVAL_DIR = Path(__file__).parent.parent / "eval"


def _chunk(source_document: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=rank,
        document_id=0,
        source_document=source_document,
        text="irrelevant for metrics tests",
        page=None,
        section_path=(),
        char_start=0,
        char_end=0,
        from_table=False,
        score=1.0,
        rank=rank,
    )


# --- fast, no I/O ---


def test_recall_at_k_true_when_expected_document_present() -> None:
    retrieved = [_chunk("/corpus/a.txt", 1), _chunk("/corpus/b.txt", 2)]
    assert recall_at_k(retrieved, ["b.txt"]) == 1.0


def test_recall_at_k_false_when_expected_document_absent() -> None:
    retrieved = [_chunk("/corpus/a.txt", 1)]
    assert recall_at_k(retrieved, ["b.txt"]) == 0.0


def test_recall_at_k_none_for_unanswerable_question() -> None:
    retrieved = [_chunk("/corpus/a.txt", 1)]
    assert recall_at_k(retrieved, []) is None


def test_reciprocal_rank_uses_first_matching_rank() -> None:
    retrieved = [_chunk("/corpus/a.txt", 1), _chunk("/corpus/b.txt", 2), _chunk("/corpus/c.txt", 3)]
    assert reciprocal_rank(retrieved, ["c.txt"]) == 1 / 3


def test_reciprocal_rank_zero_when_not_found() -> None:
    retrieved = [_chunk("/corpus/a.txt", 1)]
    assert reciprocal_rank(retrieved, ["z.txt"]) == 0.0


def test_reciprocal_rank_none_for_unanswerable_question() -> None:
    assert reciprocal_rank([_chunk("/corpus/a.txt", 1)], []) is None


def test_question_set_loads_and_has_the_expected_shape() -> None:
    questions = load_questions(EVAL_DIR / "questions.json")
    assert len(questions) == 40
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    answerable = [q for q in questions if q.expected_documents]
    unanswerable = [q for q in questions if not q.expected_documents]
    assert len(answerable) == 30
    assert len(unanswerable) == 10
    for q in unanswerable:
        assert q.category == "unanswerable"


def test_corpus_files_referenced_by_questions_all_exist() -> None:
    corpus_files = {p.name for p in (EVAL_DIR / "corpus").iterdir() if not p.name.startswith("_")}
    for q in load_questions(EVAL_DIR / "questions.json"):
        for doc in q.expected_documents:
            assert doc in corpus_files, f"{q.id} references missing corpus file {doc!r}"


# --- live: real Postgres + real embedding model ---


@requires_slow_tests
def test_eval_retrieval_finds_the_right_document_after_indexing(pg_conn) -> None:
    ensure_eval_schema(pg_conn)
    index_corpus(pg_conn, EVAL_DIR / "corpus", "structure_aware", "structure_aware")

    results = eval_search(
        pg_conn, "structure_aware-structure_aware", "What is the capital of France?", k=5, strategy="hybrid"
    )
    assert results
    assert any(r.source_document.endswith("geography_capitals.txt") for r in results)


@requires_slow_tests
def test_indexing_naive_and_structure_aware_configs_are_independent(pg_conn) -> None:
    ensure_eval_schema(pg_conn)
    naive_count = index_corpus(pg_conn, EVAL_DIR / "corpus", "naive", "naive")
    aware_count = index_corpus(pg_conn, EVAL_DIR / "corpus", "structure_aware", "structure_aware")

    # naive extraction collapses each document into one block, but naive
    # chunking still slides a fixed 1000-char window regardless of block
    # boundaries - solar_system.html's ~1071 naive-extracted chars split
    # into 2 windows, so this is 10 documents + 1 extra split, not 10 flat.
    assert naive_count == 11
    assert aware_count > naive_count

    naive_rows = pg_conn.execute(
        "SELECT count(*) FROM eval_chunks WHERE config_id = 'naive-naive'"
    ).fetchone()[0]
    aware_rows = pg_conn.execute(
        "SELECT count(*) FROM eval_chunks WHERE config_id = 'structure_aware-structure_aware'"
    ).fetchone()[0]
    assert naive_rows == naive_count
    assert aware_rows == aware_count
