from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from helpers import requires_slow_tests
from sdr.api.app import app
from sdr.config import get_settings
from sdr.storage import repository

client = TestClient(app)

UPLOAD_DIR = Path(get_settings().upload_dir)


# --- fast, no DB needed ---


def test_root_serves_the_static_ui() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Smart Document Reader" in resp.text


def test_openapi_docs_are_available() -> None:
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


# --- needs Postgres reachable, no embedding model required ---


def test_health_reports_postgres_and_pgvector(pg_conn) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["postgres"] is True
    assert data["pgvector"] is True
    assert isinstance(data["ollama"], bool)  # may be True or False depending on the environment


def test_get_missing_document_returns_404(pg_conn) -> None:
    resp = client.get("/documents/999999999")
    assert resp.status_code == 404


def test_list_documents_returns_a_list(pg_conn) -> None:
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# --- needs Postgres + the real embedding model ---


@requires_slow_tests
def test_upload_ingest_then_query_end_to_end(pg_conn) -> None:
    filename = "api_test_upload.txt"
    dest = UPLOAD_DIR / filename
    repository.delete_document(pg_conn, str(dest))

    try:
        content = b"Paris is the capital of France. It is known for the Eiffel Tower."
        resp = client.post("/documents", files={"file": (filename, content, "text/plain")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "indexed"
        assert body["chunks_indexed"] > 0
        doc_id = body["document_id"]

        resp = client.get(f"/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["source_path"] == str(dest)

        resp = client.post("/query", json={"question": "What is the capital of France?", "k": 3, "with_answer": False})
        assert resp.status_code == 200
        retrieved = resp.json()["retrieved"]
        assert any(r["source_document"] == str(dest) for r in retrieved)
        assert resp.json()["answer"] is None
    finally:
        repository.delete_document(pg_conn, str(dest))
        dest.unlink(missing_ok=True)


@requires_slow_tests
def test_reuploading_unchanged_file_is_skipped(pg_conn) -> None:
    filename = "api_test_reupload.txt"
    dest = UPLOAD_DIR / filename
    repository.delete_document(pg_conn, str(dest))

    try:
        content = b"A short unchanging test document."
        first = client.post("/documents", files={"file": (filename, content, "text/plain")})
        assert first.json()["status"] == "indexed"

        second = client.post("/documents", files={"file": (filename, content, "text/plain")})
        assert second.json()["status"] == "skipped_unchanged"
        assert second.json()["chunks_indexed"] == 0
    finally:
        repository.delete_document(pg_conn, str(dest))
        dest.unlink(missing_ok=True)


# --- needs Postgres + the real embedding model + Ollama ---


@requires_slow_tests
def test_query_with_answer_returns_a_grounded_citation(pg_conn, ollama_ready) -> None:
    filename = "api_test_answer.txt"
    dest = UPLOAD_DIR / filename
    repository.delete_document(pg_conn, str(dest))

    try:
        content = b"Paris is the capital of France. It is known for the Eiffel Tower."
        upload = client.post("/documents", files={"file": (filename, content, "text/plain")})
        assert upload.json()["status"] == "indexed"

        resp = client.post("/query", json={"question": "What is the capital of France?", "k": 3, "with_answer": True})
        assert resp.status_code == 200
        answer = resp.json()["answer"]
        assert answer is not None
        assert answer["raw_text"]
        grounded = [s for s in answer["sentences"] if s["grounded"]]
        assert grounded, f"no grounded sentence in: {answer['raw_text']!r}"
        assert grounded[0]["citations"][0]["document"] == str(dest)
    finally:
        repository.delete_document(pg_conn, str(dest))
        dest.unlink(missing_ok=True)
