from __future__ import annotations

from unittest.mock import MagicMock, patch

from sdr.embedding.model import embed_texts
from helpers import requires_slow_tests


def test_embed_texts_returns_empty_list_for_empty_input() -> None:
    assert embed_texts([]) == []


def test_embed_texts_calls_model_encode_with_normalization() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = [MagicMock(tolist=lambda: [0.1, 0.2])]

    with patch("sdr.embedding.model._get_model", return_value=fake_model):
        result = embed_texts(["hello"])

    assert result == [[0.1, 0.2]]
    _, kwargs = fake_model.encode.call_args
    assert kwargs["normalize_embeddings"] is True


@requires_slow_tests
def test_real_model_produces_sane_semantic_similarity() -> None:
    """Downloads/runs the actual configured embedding model - a genuine
    measurement, not an asserted number. Opt in with SDR_RUN_SLOW_TESTS=1."""
    import numpy as np

    vectors = embed_texts(
        [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "Quarterly revenue increased by twelve percent.",
        ]
    )
    cat, feline, revenue = (np.array(v) for v in vectors)
    sim_related = float(cat @ feline)
    sim_unrelated = float(cat @ revenue)
    assert sim_related > sim_unrelated
