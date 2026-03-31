from pathlib import Path

from backend.core.settings import load_settings


def test_load_settings_defaults(monkeypatch):
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("DEFAULT_PERFORMANCE_MODE", raising=False)
    monkeypatch.delenv("PERFORMANCE_MODE", raising=False)
    monkeypatch.delenv("OCR_ENABLED", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    settings = load_settings(Path("."))

    assert settings.app_env == "dev"
    assert settings.api_port == 8000
    assert settings.default_performance_mode is False
    assert settings.ocr_enabled is True
    assert settings.db_url.startswith("sqlite:///")
    assert settings.cors_origins == ["*"]


def test_load_settings_parses_env_values(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("PERFORMANCE_MODE", "true")
    monkeypatch.setenv("OCR_ENABLED", "false")
    monkeypatch.setenv("DB_URL", "postgresql+psycopg://user:pass@localhost:5432/obj")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    monkeypatch.setenv("MAX_OBJECTS_PER_FRAME", "77")

    settings = load_settings(Path("."))

    assert settings.app_env == "prod"
    assert settings.api_port == 9000
    assert settings.default_performance_mode is True
    assert settings.ocr_enabled is False
    assert settings.db_url.startswith("postgresql+psycopg://")
    assert settings.max_objects_per_frame == 77
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_load_settings_invalid_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("API_PORT", "not-a-number")
    settings = load_settings(Path("."))
    assert settings.api_port == 8000
