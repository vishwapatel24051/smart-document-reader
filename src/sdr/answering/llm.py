from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from sdr.config import get_settings

_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int | None
    response_tokens: int | None
    duration_seconds: float


def generate_with_stats(prompt: str) -> GenerationResult:
    """Call the local Ollama server's /api/generate and return the
    response text alongside real, Ollama-reported token counts
    (prompt_eval_count/eval_count) and wall-clock duration - used by
    Phase 7's evaluation harness for latency/token-cost columns. Raises
    httpx.HTTPError (e.g. ConnectError if Ollama isn't running) - callers
    decide how to surface that, this doesn't swallow it.
    """
    settings = get_settings()
    start = time.monotonic()
    response = httpx.post(
        f"{settings.ollama_host}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    elapsed = time.monotonic() - start
    data = response.json()
    return GenerationResult(
        text=data["response"].strip(),
        prompt_tokens=data.get("prompt_eval_count"),
        response_tokens=data.get("eval_count"),
        duration_seconds=elapsed,
    )


def generate(prompt: str) -> str:
    return generate_with_stats(prompt).text
