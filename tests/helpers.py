from __future__ import annotations

import os

import pytest

RUN_SLOW_TESTS = os.environ.get("SDR_RUN_SLOW_TESTS") == "1"

requires_slow_tests = pytest.mark.skipif(
    not RUN_SLOW_TESTS,
    reason="set SDR_RUN_SLOW_TESTS=1 to run - downloads/runs a real embedding model",
)
