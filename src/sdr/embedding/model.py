from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from sdr.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache
def _get_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured local model.

    Vectors are L2-normalized so cosine similarity reduces to a dot
    product, which is what pgvector's cosine index (Phase 4 schema) expects.
    """
    if not texts:
        return []
    vectors = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embedding_dimension() -> int:
    dim = _get_model().get_sentence_embedding_dimension()
    assert dim is not None
    return dim
