from __future__ import annotations

import httpx

from sdr.config import get_settings

_TIMEOUT_SECONDS = 120.0


def generate(prompt: str) -> str:
    """Call the local Ollama server's /api/generate. Raises httpx.HTTPError
    (e.g. ConnectError if Ollama isn't running) - callers decide how to
    surface that, this module doesn't swallow it.
    """
    settings = get_settings()
    response = httpx.post(
        f"{settings.ollama_host}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
