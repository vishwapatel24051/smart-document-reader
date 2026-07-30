from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_user: str = "sdr"
    postgres_password: str = "sdr"
    postgres_db: str = "sdr"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    log_level: str = "INFO"

    # "dense", "lexical", or "hybrid" - the default retrieval path when
    # sdr.retrieval.search() isn't given an explicit strategy. Phase 7's
    # benchmark matrix overrides this per run rather than relying on it.
    retrieval_strategy: str = "hybrid"
    retrieval_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma2:2b"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    upload_dir: str = "data/uploads"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
