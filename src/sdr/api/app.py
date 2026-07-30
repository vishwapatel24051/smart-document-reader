from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sdr.config import get_settings
from sdr.logging_setup import setup_logging

from .routes import documents, health, ingest, query

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    setup_logging(get_settings().log_level)

    app = FastAPI(
        title="Smart Document Reader",
        description="Structure-aware research assistant over a local document corpus.",
        version="0.1.0",
    )
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    # Mounted last and at "/": StaticFiles(html=True) serves index.html for
    # "/" itself but would otherwise shadow routes registered after it.
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("sdr.api.app:app", host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
