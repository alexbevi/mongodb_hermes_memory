"""Tests for config loading, schema, and persistence."""

from __future__ import annotations

import json

import pytest

from hermes_mongodb_memory import config


def test_get_config_schema_includes_required_keys():
    schema = config.get_config_schema()
    keys = {field["key"] for field in schema}
    assert {
        "connection_uri",
        "database",
        "embedding_provider",
        "embedding_api_key",
        "embedding_model",
        "embedding_dim",
        "tenant_scope",
        "auto_extract",
        "extraction_llm",
        "time_decay_half_life_days",
        "default_ttl_days",
        "hybrid_vector_weight",
        "prefetch_limit",
        "turn_ttl_days",
    } <= keys


def test_schema_marks_uri_required_and_secret():
    schema = {f["key"]: f for f in config.get_config_schema()}
    uri_field = schema["connection_uri"]
    assert uri_field["required"] is True
    assert uri_field["secret"] is True
    assert uri_field["env_var"] == "HERMES_MONGODB_URI"


def test_load_config_returns_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_MONGODB_URI", raising=False)
    cfg = config.load_config(tmp_path)
    assert cfg["database"] == "hermes_memory"
    assert cfg["embedding_provider"] == "none"
    assert cfg["connection_uri"] == ""


def test_load_config_reads_json_file(tmp_path):
    path = tmp_path / "mongodb.json"
    path.write_text(json.dumps({"database": "alt_db", "embedding_dim": 768}))
    cfg = config.load_config(tmp_path)
    assert cfg["database"] == "alt_db"
    assert cfg["embedding_dim"] == 768
    assert cfg["embedding_provider"] == "none"  # default preserved


def test_env_var_overrides_file(tmp_path, monkeypatch):
    (tmp_path / "mongodb.json").write_text(json.dumps({"connection_uri": "mongodb://from-file"}))
    monkeypatch.setenv("HERMES_MONGODB_URI", "mongodb://from-env")
    cfg = config.load_config(tmp_path)
    assert cfg["connection_uri"] == "mongodb://from-env"


def test_load_config_handles_malformed_file(tmp_path):
    (tmp_path / "mongodb.json").write_text("{not json")
    cfg = config.load_config(tmp_path)
    assert cfg == {**config.DEFAULTS}


def test_save_config_redacts_secrets(tmp_path):
    config.save_config(
        {"database": "hermes_memory", "connection_uri": "mongodb://secret", "auto_extract": True},
        tmp_path,
    )
    on_disk = json.loads((tmp_path / "mongodb.json").read_text())
    assert on_disk["database"] == "hermes_memory"
    assert on_disk["auto_extract"] is True
    assert "connection_uri" not in on_disk
    assert "embedding_api_key" not in on_disk


def test_save_config_creates_missing_directories(tmp_path):
    target = tmp_path / "nested" / "dir"
    config.save_config({"database": "x"}, target)
    assert (target / "mongodb.json").is_file()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("FALSE", False),
        ("1", True),
        (1, True),
        ("", False),
        (None, False),
    ],
)
def test_coerce_bool(raw, expected):
    assert config.coerce_bool(raw) is expected


def test_load_config_uses_hermes_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MONGODB_URI", raising=False)
    (tmp_path / "mongodb.json").write_text(json.dumps({"database": "from_home"}))
    cfg = config.load_config()
    assert cfg["database"] == "from_home"


def test_save_config_then_load_roundtrip(tmp_path):
    config.save_config({"database": "rt", "embedding_dim": 1024}, tmp_path)
    cfg = config.load_config(tmp_path)
    assert cfg["database"] == "rt"
    assert cfg["embedding_dim"] == 1024
