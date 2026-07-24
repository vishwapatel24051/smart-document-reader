from __future__ import annotations

import logging

from sdr.config import Settings, get_settings
from sdr.logging_setup import setup_logging


def test_settings_defaults_without_env_file() -> None:
    settings = Settings(_env_file=None)
    assert settings.postgres_db == "sdr"
    assert settings.database_url == "postgresql://sdr:sdr@localhost:5432/sdr"


def test_get_settings_returns_settings_instance() -> None:
    assert isinstance(get_settings(), Settings)


def test_setup_logging_configures_root_logger_level() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
