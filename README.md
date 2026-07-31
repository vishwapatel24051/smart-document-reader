# Smart Document Reader

A personal research assistant over a local document corpus.

The distinguishing claim here is **not** the RAG pipeline — those are
commodity. It's that document ingestion is treated as the primary
engineering problem, and every design choice is backed by measured numbers
rather than assertion.

The failure this addresses: most RAG tools run PDFs through a generic text
loader and split on character count. Multi-column papers get read across
columns, tables collapse into unordered numbers, scanned pages return
nothing, and captions detach from figures. Retrieval then searches corrupted
text, and the resulting bad answer gets blamed on the model.

**Benchmark report: see "Benchmark report (Phase 9)" below.** Read the
caveats there before drawing conclusions — this is a small, AI-authored
eval set (disclosed in `eval/README.md`), not an independently validated
benchmark, and its most notable finding cuts against this project's own
thesis rather than confirming it.

## Status

- [x] Phase 0 — Scaffold
- [x] Phase 1 — Document type detection
- [x] Phase 2 — Extraction router
- [x] Phase 3 — Structure-aware chunking
- [x] Phase 4 — Indexing and storage
- [x] Phase 5 — Retrieval
- [x] Phase 6 — Answering with span citations
- [x] Phase 7 — Evaluation harness
- [x] Phase 8 — API and interface
- [x] Phase 9 — Benchmark report

## Requirements

- Python 3.11+
- Docker (for Postgres + pgvector, and optionally Ollama; `docker compose up`)
- [Ollama](https://ollama.com) with a model pulled, for Phase 6 answering -
  natively installed (`ollama pull gemma2:2b`) or via the `ollama` compose
  service (`docker compose exec ollama ollama pull gemma2:2b`). Nothing
  else in the project needs it.
- Internet on first run only, to download the embedding model weights, the
  cross-encoder reranking model (if `RETRIEVAL_RERANK=true`), the LLM
  weights (`ollama pull`), and (if you don't already have them) the
  `pgvector/pgvector:pg16` / `ollama/ollama` images; everything is
  local/offline after that.

## Setup

```bash
cp .env.example .env

# Local (non-docker) dev environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest                          # fast: no DB, no real model, 9 tests skip
docker compose up -d db         # start just Postgres+pgvector for local dev
SDR_RUN_SLOW_TESTS=1 pytest     # full run: DB + real embedding model, 0 skipped
```

`SDR_RUN_SLOW_TESTS=1` opts into tests that hit a real Postgres and
download/run the real embedding model - kept opt-in so a plain `pytest`
stays fast and offline. DB-dependent tests skip individually (with a clear
reason) if Postgres isn't reachable, regardless of that flag.

## Running with Docker

```bash
docker compose up -d
docker compose exec app pytest
SDR_RUN_SLOW_TESTS=1 docker compose exec app pytest   # full run, inside the container
```

This starts Postgres with the pgvector extension (`db`), Ollama (`ollama`),
and the FastAPI app (`app`) — the query UI and API docs are at
`http://localhost:8000` (`API_HOST_PORT` to override the published port).

Running the API locally without Docker:

```bash
sdr-serve   # or: uvicorn sdr.api.app:app --reload
```

## Project layout

```
src/sdr/            application package
  config.py          env-based settings (pydantic-settings)
  logging_setup.py    logging configuration
  detection/           document-type detection from file content
  extraction/           format-appropriate extraction + quality reporting
  chunking/              structure-aware + naive-baseline chunkers
  storage/               Postgres schema, connection handling, repository
  embedding/             local embedding model wrapper
  retrieval/              dense / lexical / hybrid search + optional reranking
  answering/              LLM answer generation + span citations + groundedness
  evaluation/             matrix runner, CLI, eval-only storage/retrieval, metrics
  api/                    FastAPI app: routes, schemas, static query UI
  ingest.py              orchestrates detect -> extract -> chunk -> embed -> store
scripts/             one-off scripts (e.g. fixture generation)
eval/                evaluation corpus, question set, and results (Phase 7)
data/uploads/        files uploaded through the API (gitignored)
tests/               pytest suite
```

## Document type detection (Phase 1)

`sdr.detection.detect_file(path)` / `detect_bytes(data)` classify a document
from its **content**, never its extension — the API doesn't look at the
filename at all. It returns a `DetectionResult(doc_type, confidence, signals)`:

- `doc_type` is one of `pdf_text`, `pdf_scanned`, `docx`, `html`,
  `plain_text`, `unknown`.
- `confidence` is a heuristic 0.0–1.0 score, not a calibrated probability.
- `signals` is a tuple of human-readable evidence (e.g. which magic bytes
  matched, how many sampled PDF pages had extractable text) for debugging
  and for later phases' quality reports.

Detection never raises: unreadable, empty, or corrupt files come back as
`unknown` with low confidence and a signal explaining why, instead of an
exception. PDFs are told apart from scanned PDFs by opening them with
PyMuPDF and checking whether the first few pages yield extractable text —
not by file size or page count.

Tests live in `tests/test_detection.py` against fixtures in
`tests/fixtures/detection/`, generated by
`scripts/generate_detection_fixtures.py` (rerun it if the fixture set needs
to change). The fixtures include deliberately mislabeled files (real PDF
bytes saved with a `.txt` extension and vice versa), a corrupt PDF, a
non-DOCX zip renamed to `.docx`, and an empty file.

## Extraction router (Phase 2)

`sdr.extraction.extract(path)` detects the document (reusing Phase 1), routes
it to a format-appropriate extractor, and always returns an
`ExtractedDocument(source_path, doc_type, blocks, quality)` — never raises.
Each `ExtractedBlock` carries `block_type` (`heading` / `paragraph` /
`table`), `order` (reading-order index), `section_path` (heading hierarchy),
`page`, `from_ocr`, and, for tables, `table_rows` (structured, not flattened)
and `caption`.

- **PDF reading order**: blocks are ordered by detecting a vertical "gutter"
  (a strip near the page's horizontal center that no block crosses). If one
  exists and there's content on both sides, the page is treated as two
  columns and sorted column-by-column, top-to-bottom; otherwise it falls
  back to a plain top-to-bottom sort. This is a structural heuristic, not a
  layout model — it's verified against a synthetic two-column fixture, not
  against real papers yet.
- **PDF tables**: PyMuPDF's built-in `find_tables()` (ruling-line based).
  Rows are kept structured (`table_rows`); text under a table's bounding box
  is excluded from surrounding paragraph blocks so it isn't duplicated.
- **PDF headings**: a block is treated as a heading if its dominant font
  size is ≥1.15× the document's median span size and it's short — a
  font-size heuristic, not real style/outline data (PDFs don't reliably
  expose either).
- **Captions**: a paragraph immediately preceding a table and matching
  `^(table|figure|fig\.)\s*\d+` is detached from the paragraph stream and
  attached as the table's `caption` instead of kept as an unrelated floating
  block. HTML tables prefer a native `<caption>` tag first.
- **OCR fallback**: any page (regardless of the document-level detected
  type) that yields fewer than 20 recovered characters is rendered to an
  image and sent through pytesseract. If tesseract isn't installed, or OCR
  finds nothing, that's recorded in `quality.failures` — never silently
  swallowed. On this dev machine tesseract isn't installed, so
  `tests/test_extraction.py`'s OCR test asserts the honest "attempted,
  unavailable" outcome; the Docker image installs `tesseract-ocr` so the
  real recovery path runs there.
- **DOCX / HTML**: heading hierarchy from native styles (`Heading N`) /
  tags (`h1`–`h6`); tables via native table structures; same caption
  handling as PDF.

**Deviation from the Phase 0 plan**: PDF tables use PyMuPDF's native
`find_tables()` instead of the originally stated pdfplumber, since it
already ships with the PyMuPDF dependency Phase 1 added and tested out
reliably on ruled tables — one fewer dependency for the same result. Flagging
this since it wasn't asked about first.

Tests live in `tests/test_extraction.py` against fixtures in
`tests/fixtures/extraction/`, generated by
`scripts/generate_extraction_fixtures.py`. Fixtures include a two-column PDF
(constructed so a naive top-to-bottom sort would visibly interleave the
columns wrong), a captioned ruled-line table PDF, an image-only PDF with
real rendered text (for the OCR path), and matching structured DOCX/HTML
fixtures.

## Chunking (Phase 3)

`sdr.chunking` turns an `ExtractedDocument`'s blocks into `Chunk`s
(`text`, `source_document`, `page`, `section_path`, `char_start`/`char_end`,
`from_table`, `chunk_index`, `strategy`). Both chunkers below key their char
spans off the same canonical flattened-document text
(`flatten_document()`), so their spans are directly comparable.

- **`chunk_structure_aware(document, max_chars=1200)`**: chunks on semantic
  boundaries. A table is always its own chunk and is never merged with
  surrounding text or split. A heading always starts a new chunk. Paragraphs
  accumulate into the current chunk until adding the next one would exceed
  `max_chars` — a soft target, not a hard cutoff: a single paragraph or
  table bigger than the budget still becomes one chunk rather than being
  split mid-unit. A heading immediately followed by a table (nothing to
  pair it with) is dropped rather than emitted as a content-free chunk;
  section metadata still reaches the table via its own extraction-time
  `section_path`.
- **`chunk_naive(document, chunk_size=1000, overlap=100)`**: the
  deliberately naive baseline the spec calls for — a fixed-size sliding
  window over the flattened text with zero awareness of paragraphs, tables,
  or sections. Its metadata (page/section/`from_table`) is a best-effort
  guess (whichever source block the window overlaps most), which is
  approximate by construction. This isn't a fallback path; it exists so
  Phase 7 can measure the structure-aware chunker against something.

`max_chars` / `chunk_size` are unvalidated defaults, not tuned numbers —
no chunk-size sweep has been run. Tests
(`tests/test_chunking.py`) reuse Phase 2's fixtures directly (no new
binary fixtures needed) and include a concrete demonstration of the
failure mode this project is about: `table.pdf`'s table survives whole
under `chunk_structure_aware` but gets split under `chunk_naive`.

## Indexing and storage (Phase 4)

`sdr.ingest.ingest_document(path, conn)` runs detect → extract → chunk
(structure-aware) → embed → store, and returns an `IngestOutcome(status,
document_id, chunks_indexed, reason)` where `status` is `indexed`,
`skipped_unchanged`, or `failed`. It never raises.

- **Incremental indexing**: each document's raw bytes are SHA-256 hashed.
  If the stored hash for that `source_path` already matches, ingestion
  skips immediately — no extraction, chunking, embedding, or writes.
  Re-ingesting a *changed* document only deletes and reinserts *that
  document's* chunk rows (`repository.replace_chunks`, scoped by
  `document_id`); no other document's rows are touched. Verified directly:
  `tests/test_ingest.py::test_reingesting_a_changed_document_does_not_touch_other_documents`
  ingests two documents, changes one, and asserts the other's chunk count
  is unchanged after re-ingesting.
- **Schema** (`sdr/storage/schema.sql`): `documents` (one row per source
  file, content hash, extraction quality fields) and `chunks` (one row per
  chunk, `vector(384)` embedding column, generated `tsvector` column for
  lexical search). Applied idempotently (`CREATE TABLE IF NOT EXISTS` etc.)
  on every connection via `ensure_schema()` — no separate migration step
  or tool for this project's scope.
- **Dense embeddings**: `sentence-transformers`, local model
  (`BAAI/bge-small-en-v1.5`, 384-dim, overridable via `EMBEDDING_MODEL` —
  changing it means changing the schema's `vector(384)` column too, they're
  coupled). Vectors are L2-normalized so pgvector's cosine index applies
  cleanly. **First use downloads the model weights from Hugging Face** (a
  few hundred MB) — the only point in the ingestion/retrieval path that
  needs internet; fully offline afterward, cached under the container/host
  Hugging Face cache.
- **Lexical index**: Postgres `tsvector` (generated column) + GIN index,
  decided over a Python BM25 library or a specialized extension — it's
  persisted, updates incrementally with no extra code, and needs no new
  dependency. **Named honestly, not as BM25**: ranking will use Postgres's
  own `ts_rank_cd`, not the Okapi BM25 formula. This project still calls
  the concept "lexical search" rather than "BM25" wherever it means this.
- **Why Postgres tsvector over rank_bm25 or a BM25 Postgres extension**:
  this was a real fork, and I asked before building — Postgres tsvector
  was chosen for being persisted and incrementally maintained by Postgres
  itself, at zero extra operational cost for a solo local project. A true
  BM25 library remains a documented option if retrieval quality later
  demands it.

**Fully verified against a live database**, not just unit-tested against
mocks: Postgres wasn't reachable in this environment by default (Docker
daemon wasn't running, and a docker-compose port conflict I hit along the
way, see below), so I started Docker, brought up the `db` service, and ran
every DB- and model-dependent test for real — 45/45 passed, zero skipped.
Two real bugs were only caught this way:
1. `register_vector()` (the pgvector type adapter) was being called before
   `ensure_schema()` — it failed on a fresh database because the `vector`
   type doesn't exist until `CREATE EXTENSION` has run. Fixed by ensuring
   schema first, both in `db.get_connection()` and in the test fixture.
2. The default port mapping (`5432:5432`) collided with an unrelated,
   already-running native Postgres install on this machine. Since that's a
   common situation for anyone with a local Postgres, the fix is a general
   one: `docker-compose.yml` now maps the `db` service to host port 5433 by
   default (`POSTGRES_HOST_PORT` to override), and `.env.example` matches.
   Internal app↔db traffic inside docker compose is unaffected either way.

Tests: `tests/test_storage.py` (schema, upsert/lookup, chunk-replacement
isolation — DB only, no embedding model needed) and `tests/test_ingest.py`
/ the real-model case in `tests/test_embedding.py` (full pipeline — DB
*and* the real model). Both gate gracefully: `tests/conftest.py`'s `pg_conn`
fixture skips with a clear message if Postgres isn't reachable, and the
model-dependent tests are behind `SDR_RUN_SLOW_TESTS=1` so a plain `pytest`
run stays fast and offline by default.

## Retrieval (Phase 5)

`sdr.retrieval.search(conn, query, k=10, strategy=None, rerank=None)` is
the one interface dense, lexical, and hybrid retrieval sit behind —
`strategy` defaults to `settings.retrieval_strategy` ("hybrid") and
`rerank` to `settings.retrieval_rerank` (off), but both can be overridden
per call, which is what Phase 7's benchmark matrix will do to run the same
query through every configuration. Every path returns
`list[RetrievedChunk]` (`chunk_id`, `document_id`, `source_document`,
`text`, `page`, `section_path`, `char_start`/`char_end`, `from_table`,
`score`, `rank`) — the same shape regardless of strategy.

- **`search_dense`**: embeds the query with the same local model used at
  ingest time, orders chunks by pgvector cosine distance (`<=>`).
- **`search_lexical`**: Postgres `websearch_to_tsquery` (tolerant of
  arbitrary user input, never raises on unusual characters) against the
  `tsv` column, ranked by `ts_rank_cd` — again, Postgres's own ranking
  function, not Okapi BM25 (see Phase 4).
- **`search_hybrid`**: fuses dense + lexical via Reciprocal Rank Fusion
  (`score += 1 / (60 + rank)` per list, `60` being the standard constant
  from the original RRF paper, not tuned here). RRF combines by *rank*, not
  raw score, specifically because cosine similarity and `ts_rank_cd` are on
  incomparable scales — fusing by score would need a normalization scheme
  neither retriever's numbers are designed for.
- **`rerank(query, candidates, top_k)`**: optional second-stage
  cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default) that
  scores query/chunk pairs jointly — more accurate than independent
  dense/lexical scoring, too slow to run over a whole corpus, so it only
  reorders a small candidate pool. Behind the `rerank` flag/setting, per
  the spec.

**Real bug caught during live verification** (not by a mock): a bare
`c.embedding <=> %(query_vector)s` comparison silently sent the Python
embedding list as `float8[]`, not `vector`, and Postgres had no matching
`<=>` overload — `UndefinedFunction`. This didn't show up in Phase 4
because an `INSERT`'s target column type disambiguates the parameter type
server-side; a bare comparison in a `WHERE`/`ORDER BY` doesn't give
psycopg's client-side type adapter the same hint. Fixed with an explicit
`%(query_vector)s::vector` cast in `dense.py`'s SQL.

**Verified live**, again: `tests/test_retrieval.py` ingests a small,
deliberately distinct 3-document corpus (cats / finance / weather) and
checks concrete, falsifiable behavior rather than just "it returns
something" — lexical search finds an exact keyword match; dense search
finds a paraphrase with **zero literal word overlap** with the target
document; lexical search returns **nothing** for that same paraphrase
(demonstrating exactly the gap hybrid exists to close); hybrid surfaces the
dense-only match; reranking actually reorders a real candidate pool with
a real cross-encoder. All 8 tests passed against live Postgres + real
models, gated behind `SDR_RUN_SLOW_TESTS=1` like Phase 4's.

## Answering with span citations (Phase 6)

`sdr.answering.answer_query(conn, query, k=5, strategy=None, rerank=None,
groundedness_threshold=0.4)` retrieves chunks (Phase 5), generates an
answer with a local LLM via Ollama, splits it into sentences, and attaches
a structured `Citation(document, page, char_start, char_end, chunk_id)` to
each sentence that's lexically supported by a retrieved chunk. Returns an
`Answer(query, sentences, raw_text, retrieved_chunk_ids)` — never raises on
empty retrieval (returns an `Answer` with no sentences), but does let
`httpx.HTTPError` propagate if Ollama isn't reachable, rather than
pretending an answer was generated.

**The groundedness check is a lexical-overlap heuristic, not verified
hallucination detection — stated as plainly as possible in both the code
and here.** `score_sentence(sentence, chunk_text)` is the fraction of a
sentence's distinct words that also appear in a chunk's text. A sentence
clears the groundedness threshold (default 0.4) and gets a citation for
every chunk it overlaps that much with, capped at 2. This means:

- A sentence can score high while asserting something the chunk doesn't
  actually say (e.g. negating it, or stitching together two unrelated facts
  that both happen to appear in the same chunk).
- A sentence can score low while being a faithful, accurate rewording that
  just uses different vocabulary than the source.

"Grounded" here means "shares vocabulary with a retrieved passage," not
"is factually correct." Nothing in this project's code, tests, or output
should describe it otherwise.

**Verified live**, same pattern as Phases 4/5: Ollama was already installed
and running natively on this dev machine (`gemma2:2b`, 1.6GB, already
pulled) — I used it directly rather than also spinning up the containerized
`ollama` service. Asked "What is the capital of France?" against a small
ingested document, the real pipeline produced `"Paris is the capital of
France."` as a single sentence, scored 1.00 (full word overlap) and cited
back to the correct source document and the chunk's exact char span. Asked
a question the context couldn't answer, the model itself said so directly
("The provided text does not contain information about...") and the
groundedness check independently scored that sentence 0.29 — below
threshold, `grounded=False` — for an unrelated reason (different
vocabulary), which happens to agree with the model's own admission here but
isn't the same check.
`tests/test_answering.py` has 8 fast unit tests (sentence splitting,
overlap scoring, prompt construction, the empty-retrieval path) plus one
live end-to-end test gated behind `SDR_RUN_SLOW_TESTS=1` and a new
`ollama_ready` fixture (skips with a clear reason if Ollama isn't
reachable, same pattern as the Postgres `pg_conn` fixture).

`docker-compose.yml` now also has an `ollama` service (image
`ollama/ollama`) for a fully self-contained `docker compose up`, not
published to a fixed host port by default (`OLLAMA_HOST_PORT` to override)
for the same reason as the Postgres port: Ollama is commonly already
running natively on 11434.

## Evaluation harness (Phase 7)

**Read `eval/README.md` first.** The corpus (10 documents) and question set
(40 questions) in `eval/` were authored by Claude, not independently
reviewed by a human — the spec for this phase explicitly requires that
disclosure ("do not generate questions with an LLM and treat them as ground
truth without saying so"), and `eval/README.md` is it, in full.

`sdr.evaluation` builds and runs a config matrix: **extraction**
(`naive` — the generic-text-loader baseline described in this README's
opening pitch, added this phase via `sdr.extraction.naive_extract` — vs.
`structure_aware`, Phase 2's `extract`) × **chunking** (`naive` vs.
`structure_aware`, Phase 3) × **retrieval** (`dense` vs. `hybrid`) ×
**rerank** (on/off) — 16 configurations. Each `(extraction, chunking)` pair
gets its own isolated index in a dedicated `eval_chunks` table (not the
production `documents`/`chunks` tables — re-indexing the same 10 documents
four different ways would collide with the production schema's
one-row-per-path model), scoped by a `config_id` column so runs never leak
into each other.

```bash
python -m sdr.evaluation.cli --k 5                    # retrieval only: recall@k, MRR, latency
python -m sdr.evaluation.cli --k 5 --with-answers      # + groundedness rate (needs Ollama, ~14 min for the full matrix)
```

Writes a timestamped CSV to `eval/results/` (and `latest.csv`), and prints
a table to stdout.

### What was actually measured, run on this dev machine (Apple Silicon Mac, `gemma2:2b`, `BAAI/bge-small-en-v1.5`, no GPU)

**Retrieval metrics (recall@5, MRR) are saturated at 1.00 across all 16
configurations.** This is a real, honest finding, not a bug — this project's
eval corpus has only 10 documents, each on a completely distinct topic, so
"is the right document anywhere in the top 5 (or even top 1 — same result
at `--k 1`)" turned out to be too easy a bar for *any* configuration,
including the naive/naive baseline, to fail. **This means the retrieval
metrics on this specific corpus do not currently discriminate between naive
and structure-aware extraction or chunking** — the corpus would need
documents that are topically similar to each other (so mangled text from
extraction/chunking failures could plausibly outrank the genuinely relevant
document) to actually stress-test that. Recorded here instead of hidden:
this is exactly the kind of negative/inconclusive result the project's
own rules require reporting.

Latency did show real, if modest, differences: reranking roughly doubles
p50 latency (~18ms → ~35-50ms) across every configuration, as expected
for an extra cross-encoder pass; retrieval strategy and extraction/chunking
choice made only a few milliseconds of difference at this corpus size —
unsurprising given the whole corpus is under 100 chunks in every
configuration.

### `--with-answers` results (groundedness rate, unanswerable decline rate)

Full run: 40 questions × 16 configurations × real `gemma2:2b` generation,
~15 minutes wall time. Full table in `eval/results/latest.csv`; summarized
here.

**This did not confirm the hypothesis it was built to test, and that's
reported as measured, not adjusted.** The expectation going in was that
structure-aware extraction/chunking would show a *higher* groundedness rate
than naive (better-preserved tables and reading order → answers that more
clearly echo the source text). What was measured is close to the opposite:

| extraction × chunking | groundedness rate (range across retrieval/rerank) | unanswerable decline rate (range) |
|---|---|---|
| naive × naive | 0.80 – 0.90 | 0.40 – 0.80 |
| naive × structure_aware | 0.80 – 0.85 | 0.60 – 0.80 |
| structure_aware × naive | 0.825 – 0.90 | 0.40 – 0.70 |
| structure_aware × structure_aware | 0.775 – 0.80 | 0.80 – 0.90 |

`structure_aware`×`structure_aware` — the configuration this whole project
argues for — has the **lowest** groundedness rate of the four extraction ×
chunking pairs, and the **highest** unanswerable-decline rate (best at
correctly refusing to answer the 10 unanswerable questions). Those two
numbers move together, and a plausible mechanical reason why: naive
chunking produces one large chunk per document (often the entire
document), so an answer sentence has a big pool of the document's own
vocabulary to lexically overlap with, even when the LLM is combining or
slightly misstating facts. Structure-aware chunking produces many small,
specific chunks — an answer sentence only overlaps well with the one exact
chunk it was actually drawn from, so the groundedness heuristic (Phase 6:
lexical overlap, not semantic entailment) scores it more strictly. In other
words, this may be measuring "structure-aware chunking makes the lexical-
overlap heuristic more conservative," not "structure-aware chunking
produces less accurate answers" — but this evaluation, as built, cannot
tell those two apart, because groundedness here is defined by that same
heuristic, and nothing here independently checked whether the *answers*
generated were more or less factually correct.

Reranking showed no consistent directional effect on groundedness across
the 16 configurations (moved up in some, down in others, always by ≤0.05)
— its only consistent, measured effect was on latency (below). Dense
retrieval's groundedness rate was equal to or slightly higher than
hybrid's in every one of the four extraction×chunking pairs (never lower),
a small but consistent enough pattern to note, though not a large enough
gap or sample (40 questions, one run) to treat as conclusive.

Latency: reranking consistently roughly doubles retrieval p50 (e.g.
naive×naive dense: 48ms → 83ms); `structure_aware`×`structure_aware`
without reranking was the fastest configuration measured (~24ms p50) despite
producing the most chunks, likely because its chunks are shorter on average
so cosine/text-search comparisons are cheaper — not confirmed by a direct
profiling breakdown, noted as a hypothesis. Average response length was
flat across all configurations (~19–21 tokens per answer) — the model
gives similarly short answers regardless of what it's shown.

**Caveats, stated plainly**: this is one run, no repeated trials, 40
questions total (30 graded for recall/groundedness, 10 for unanswerable
decline) against a 10-document corpus small enough that retrieval itself
is saturated (see above). Differences of a few percentage points here are
within plausible single-run noise for a sample this size. This result is
reported because measuring and reporting it honestly — including when it
cuts against the project's own thesis — is what this evaluation harness
exists to do, not because it's a confident, generalizable finding about
structure-aware extraction.

## API and interface (Phase 8)

FastAPI app (`sdr/api/`) exposing the pipeline built in Phases 4-6, plus a
minimal static query UI. OpenAPI docs are automatic at `/docs`
(interactive) and `/openapi.json`.

| Endpoint | What it does |
|---|---|
| `GET /health` | Checks Postgres, pgvector, and Ollama reachability; never raises even if they're down. |
| `POST /documents` | Multipart file upload → saved under `UPLOAD_DIR` (stable path by filename) → `sdr.ingest.ingest_document()`. Re-uploading an unchanged file returns `status: "skipped_unchanged"` — incremental indexing works through the API the same as it does calling the pipeline directly. |
| `GET /documents` / `GET /documents/{id}` | Document status: extraction quality (pages, OCR, tables, failures), content hash, timestamps. 404 for an unknown id. |
| `POST /query` | `{question, k, strategy, rerank, with_answer}` → retrieved chunks, and (if `with_answer`) a generated answer with per-sentence groundedness and citations — the same `sdr.retrieval.search()` / `sdr.answering.answer_query()` used everywhere else in the project, not reimplemented for the API. |

The query UI (`sdr/api/static/index.html`) is intentionally minimal, per
the spec: a question box, k/strategy/rerank/answer controls, and a results
view showing the answer (each sentence marked grounded/ungrounded with its
citations) alongside the raw retrieved chunks. No build step, no framework,
one static HTML file with vanilla JS `fetch`.

**Verified in an actual browser, not just via TestClient**: started the
real server (`sdr-serve`), used Playwright/Chromium headless to load the
page, uploaded `mount_everest.pdf` through the API, and submitted "Who
were the first two people to summit Mount Everest?" through the UI form.
Zero console errors; the rendered page showed the correct grounded answer
("...Edmund Hillary of New Zealand and Tenzing Norgay, a Sherpa
mountaineer from Nepal", grounded 1.00) with an accurate citation
(`mount_everest.pdf, p.1`) and the matching retrieved chunk text below it.

`docker-compose.yml`'s `app` service now runs `sdr-serve` (replacing the
Phase 0-7 `sleep infinity` placeholder) and publishes port 8000
(`API_HOST_PORT` to override — 8000 had no existing conflict on this dev
machine, unlike Postgres/Ollama's default ports).

Tests: `tests/test_api.py`, via FastAPI's `TestClient` (no separate server
process needed) — static UI serving and OpenAPI docs run unconditionally;
health/document-listing need Postgres (`pg_conn`); upload→ingest→query and
the with-answer path need the real embedding model and, for the latter,
Ollama (`SDR_RUN_SLOW_TESTS=1`, `ollama_ready`), same gating pattern as
every other live-dependent test in this project.

## Benchmark report (Phase 9)

Full 16-configuration matrix run: extraction (`naive` vs `structure_aware`)
× chunking (`naive` vs `structure_aware`) × retrieval (`dense` vs
`hybrid`) × rerank (on/off). All numbers below are from a single real run
— raw data in `eval/results/latest.csv` — nothing here is estimated or
back-filled.

**Read `eval/README.md` before trusting any of this.** The corpus and
question set were authored by Claude (this project's own AI assistant),
not independently reviewed by a human. That disclosure is load-bearing,
not a footnote: it's why the retrieval metrics below are saturated
(uninformative) and why the groundedness numbers should be read as "what
this heuristic measured on this small AI-authored set," not as a
validated claim about answer quality.

**Corpus**: `eval/corpus/`, 10 documents (2 plain text, 3 HTML, 2 DOCX,
3 PDF — one multi-column, one ruled-table, one flowing text), spanning
geography, science, history, a programming language, and two documents
with deliberately synthetic/fictional data. Full manifest in
`eval/README.md`. **Questions**: `eval/questions.json`, 40 questions (30
answerable, tied to a specific expected source document; 10 deliberately
unanswerable from this corpus).

**Hardware**: Apple M1 Pro, 10 cores, 16 GB RAM, macOS 26.5.2. No GPU used
— all inference (embedding, reranking, LLM generation) ran on CPU.

**Software**: Python 3.14.2 · PostgreSQL 16.14 + pgvector 0.8.5 ·
PyMuPDF 1.28.0 · sentence-transformers 5.6.1 · psycopg 3.3.4 · FastAPI
0.141.1. **Models**: embeddings `BAAI/bge-small-en-v1.5` (384-dim) ·
reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · answering LLM
`gemma2:2b` via Ollama (1.6 GB, CPU inference).

### Full results

`k=5`. `grounded` / `unans. OK` / `avg tokens` come from the `--with-answers`
pass (real `gemma2:2b` calls, one per question per configuration — 640
generations total). `recall@k` and `MRR` are identical (1.00) in every row;
see below for why that's a real, reported limitation, not an omission.

| extraction | chunking | retrieval | rerank | recall@k | MRR | p50 ms | p95 ms | grounded | unans. OK | avg tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | naive | dense | False | 1.00 | 1.00 | 48 | 68 | 0.900 | 0.40 | 19.6 |
| naive | naive | dense | True | 1.00 | 1.00 | 83 | 138 | 0.850 | 0.60 | 21.1 |
| naive | naive | hybrid | False | 1.00 | 1.00 | 50 | 79 | 0.800 | 0.80 | 20.3 |
| naive | naive | hybrid | True | 1.00 | 1.00 | 83 | 112 | 0.850 | 0.60 | 21.3 |
| naive | structure_aware | dense | False | 1.00 | 1.00 | 47 | 63 | 0.850 | 0.60 | 19.9 |
| naive | structure_aware | dense | True | 1.00 | 1.00 | 81 | 118 | 0.825 | 0.70 | 21.2 |
| naive | structure_aware | hybrid | False | 1.00 | 1.00 | 49 | 73 | 0.800 | 0.80 | 20.9 |
| naive | structure_aware | hybrid | True | 1.00 | 1.00 | 83 | 249 | 0.825 | 0.70 | 20.6 |
| structure_aware | naive | dense | False | 1.00 | 1.00 | 50 | 63 | 0.850 | 0.60 | 19.8 |
| **structure_aware** | **naive** | **dense** | **True** | **1.00** | **1.00** | **90** | **188** | **0.900** | **0.40** | **20.2** |
| structure_aware | naive | hybrid | False | 1.00 | 1.00 | 50 | 64 | 0.850 | 0.60 | 19.9 |
| structure_aware | naive | hybrid | True | 1.00 | 1.00 | 86 | 108 | 0.825 | 0.70 | 20.2 |
| **structure_aware** | **structure_aware** | **dense** | **False** | 1.00 | 1.00 | **24** | 78 | **0.800** | 0.80 | 20.2 |
| structure_aware | structure_aware | dense | True | 1.00 | 1.00 | 91 | 146 | **0.775** | **0.90** | 18.9 |
| structure_aware | structure_aware | hybrid | False | 1.00 | 1.00 | **25** | 72 | **0.775** | **0.90** | 19.1 |
| structure_aware | structure_aware | hybrid | True | 1.00 | 1.00 | 70 | 98 | **0.775** | **0.90** | 18.7 |

### What performed worse — stated directly, per the spec's own requirement

**Retrieval quality (recall@5, MRR) does not discriminate between any of
these configurations — including the naive/naive baseline.** All 16 score
a perfect 1.00. This was checked, not assumed: the same saturation held at
`k=1`. With only 10 documents, each on a completely distinct topic,
"is the right document anywhere in the top-k" was too easy a bar for any
configuration to fail. This is this report's most direct limitation: on
this corpus, at this metric, the entire premise of the project — that
structure-aware extraction retrieves better — **is untested**, not
confirmed. A corpus with topically-overlapping documents (so a
badly-extracted competitor could plausibly outrank the right one) would be
needed to actually stress this.

**`structure_aware`×`structure_aware` — the configuration this project
argues for — has the *lowest* groundedness rate of the four extraction ×
chunking pairs (0.775–0.800, vs. up to 0.900 for `naive`×`naive` and
`structure_aware`×`naive`) and correspondingly the *highest*
unanswerable-decline rate (0.80–0.90).** The likely mechanism (explained
in the Phase 7 section above): naive chunking produces one large chunk per
document, giving the lexical-overlap groundedness heuristic a bigger
vocabulary pool to match against, even for a loosely-accurate answer.
Structure-aware chunking's smaller, specific chunks make that same
heuristic stricter. This may be "the heuristic got more conservative," not
"the answers got worse" — this evaluation cannot distinguish those two
explanations, because groundedness here *is* the heuristic. Recorded as
measured, not adjusted to match the expected direction.

**Reranking consistently costs latency for no consistent groundedness
benefit.** p50 roughly doubles in every one of the 8 dense/hybrid pairs
it's toggled on for (e.g. naive×naive dense: 48ms → 83ms; the worst case,
naive×structure_aware hybrid, hit 249ms p95). Its effect on groundedness
moved in both directions across the 16 rows, never by more than 0.05 — not
distinguishable from noise at this sample size (40 questions, one run).

**Fastest configuration measured**: `structure_aware`×`structure_aware`
without reranking (24–25ms p50) — despite producing the most chunks of any
configuration, plausibly because those chunks are shorter on average
(not confirmed by direct profiling, noted as a hypothesis).

### Caveats that apply to every number above

One run, no repeated trials, 40 questions (30 graded for recall/MRR/
groundedness, 10 for unanswerable-decline), against a corpus small enough
that retrieval itself is saturated. Differences of a few percentage points
in the groundedness/decline columns are within plausible single-run noise
for a sample this size — treat the *direction* (structure-aware chunking
trending stricter) as the finding, not the exact decimal values. This
report exists to show the comparison, including where it doesn't flatter
the project's own thesis, per this phase's explicit instruction — not to
claim a validated, generalizable result.
