from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

from sdr.storage import get_connection

from .runner import ConfigResult, load_questions, run_matrix
from .schema import ensure_eval_schema

_REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = _REPO_ROOT / "eval" / "corpus"
QUESTIONS_PATH = _REPO_ROOT / "eval" / "questions.json"
RESULTS_DIR = _REPO_ROOT / "eval" / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 7 evaluation matrix.")
    parser.add_argument("--k", type=int, default=5, help="top-k for retrieval (default: 5)")
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="also run answer_query() and compute groundedness rate - slow, needs Ollama running",
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")
    print(f"Corpus: {args.corpus}")
    print(f"with_answers={args.with_answers}  k={args.k}\n")

    with get_connection() as conn:
        ensure_eval_schema(conn)
        results = run_matrix(
            conn, args.corpus, questions, k=args.k, with_answers=args.with_answers, on_progress=print
        )

    print()
    _print_table(results)
    out_path = _write_csv(results)
    print(f"\nWrote {out_path}")
    return 0


def _print_table(results: list[ConfigResult]) -> None:
    has_answers = any(r.groundedness_rate is not None for r in results)
    headers = ["extraction", "chunking", "retrieval", "rerank", "recall@k", "MRR", "p50 ms", "p95 ms"]
    if has_answers:
        headers += ["grounded", "unanswer. OK", "avg resp tok"]

    rows: list[list[str]] = []
    for r in results:
        row = [
            r.extraction,
            r.chunking,
            r.retrieval,
            str(r.rerank),
            f"{r.recall_at_k:.2f}",
            f"{r.mrr:.2f}",
            f"{r.latency_p50_ms:.0f}",
            f"{r.latency_p95_ms:.0f}",
        ]
        if has_answers:
            row += [
                f"{r.groundedness_rate:.2f}" if r.groundedness_rate is not None else "-",
                f"{r.unanswerable_decline_rate:.2f}" if r.unanswerable_decline_rate is not None else "-",
                f"{r.avg_response_tokens:.0f}" if r.avg_response_tokens is not None else "-",
            ]
        rows.append(row)

    widths = [max(len(str(h)), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def _write_csv(results: list[ConfigResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"results_{timestamp}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "extraction",
                "chunking",
                "retrieval",
                "rerank",
                "n_questions",
                "n_graded",
                "recall_at_k",
                "mrr",
                "latency_p50_ms",
                "latency_p95_ms",
                "groundedness_rate",
                "unanswerable_decline_rate",
                "avg_response_tokens",
                "total_wall_seconds",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.extraction,
                    r.chunking,
                    r.retrieval,
                    r.rerank,
                    r.n_questions,
                    r.n_graded,
                    r.recall_at_k,
                    r.mrr,
                    r.latency_p50_ms,
                    r.latency_p95_ms,
                    r.groundedness_rate,
                    r.unanswerable_decline_rate,
                    r.avg_response_tokens,
                    r.total_wall_seconds,
                ]
            )
    latest_path = RESULTS_DIR / "latest.csv"
    latest_path.write_bytes(out_path.read_bytes())
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
