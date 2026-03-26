"""
tests/test_config.py
─────────────────────
Unit tests for the Settings configuration module.
"""

import pytest
from src.config import get_settings, Settings


def test_settings_singleton():
    """get_settings() should return the same instance each time."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_default_values():
    """Default values should match expected production defaults."""
    settings = Settings()
    assert settings.top_k == 5
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.search_type == "mmr"
    assert settings.max_new_tokens == 512
    assert "MiniLM" in settings.embedding_model
    assert "Mistral" in settings.repo_id


def test_index_dir_default():
    """Default index directory should be 'faiss_index'."""
    settings = Settings()
    assert settings.index_dir == "faiss_index"


def test_env_override(monkeypatch):
    """Environment variables should override defaults."""
    monkeypatch.setenv("TOP_K", "10")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    s = Settings()
    assert s.top_k == 10
    assert s.chunk_size == 1000
