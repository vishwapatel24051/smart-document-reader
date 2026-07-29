from __future__ import annotations

from sdr.answering import answer, score_sentence, split_sentences
from sdr.answering.groundedness import ground_sentence
from sdr.answering.prompt import build_prompt
from sdr.retrieval import RetrievedChunk
from helpers import requires_slow_tests


def _chunk(chunk_id: int, text: str, **overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=chunk_id,
        document_id=1,
        source_document="/tmp/doc.txt",
        text=text,
        page=None,
        section_path=(),
        char_start=0,
        char_end=len(text),
        from_table=False,
        score=1.0,
        rank=1,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


# --- fast, no I/O ---


def test_split_sentences_splits_on_terminal_punctuation() -> None:
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_sentences_empty_text_returns_no_sentences() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_score_sentence_full_word_overlap_scores_one() -> None:
    assert score_sentence("cats hunt mice", "cats hunt mice at night") == 1.0


def test_score_sentence_no_overlap_scores_zero() -> None:
    assert score_sentence("cats hunt mice", "quarterly revenue increased") == 0.0


def test_ground_sentence_attaches_citation_above_threshold() -> None:
    chunk = _chunk(1, "Domestic cats are small carnivorous mammals.")
    result = ground_sentence("Cats are small carnivorous mammals.", [chunk], threshold=0.4)
    assert result.grounded is True
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == 1


def test_ground_sentence_no_citation_below_threshold() -> None:
    chunk = _chunk(1, "Quarterly revenue increased this fiscal year.")
    result = ground_sentence("Cats are small carnivorous mammals.", [chunk], threshold=0.4)
    assert result.grounded is False
    assert result.citations == ()


def test_build_prompt_includes_query_and_context_text() -> None:
    chunk = _chunk(1, "Paris is the capital of France.", source_document="geo.txt", page=3)
    prompt = build_prompt("What is the capital of France?", [chunk])
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital of France." in prompt
    assert "geo.txt" in prompt
    assert "page 3" in prompt


def test_answer_query_returns_empty_answer_when_nothing_retrieved(monkeypatch) -> None:
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: [])
    result = answer.answer_query(conn=None, query="anything")  # conn unused on the empty-retrieval path
    assert result.sentences == ()
    assert result.raw_text == ""
    assert result.retrieved_chunk_ids == ()


# --- live: real Postgres + real embedding model + real Ollama ---


@requires_slow_tests
def test_answer_query_produces_a_grounded_citation_from_a_real_corpus(pg_conn, ollama_ready, tmp_path) -> None:
    from sdr.ingest import ingest_document
    from sdr.storage import repository

    source = tmp_path / "geography.txt"
    source.write_text(
        "Paris is the capital of France. It is known for the Eiffel Tower "
        "and the Louvre museum."
    )
    repository.delete_document(pg_conn, str(source))
    outcome = ingest_document(source, pg_conn)
    assert outcome.status == "indexed"

    try:
        result = answer.answer_query(pg_conn, "What is the capital of France?", k=3)

        assert result.raw_text
        assert result.sentences
        assert result.retrieved_chunk_ids

        grounded_sentences = [s for s in result.sentences if s.grounded]
        assert grounded_sentences, f"no grounded sentence in: {result.raw_text!r}"
        first_citation = grounded_sentences[0].citations[0]
        assert first_citation.document == str(source)
    finally:
        repository.delete_document(pg_conn, str(source))
