"""Configuration loading, schema, and persistence for the MongoDB plugin.

Config flows: env vars > ``$HERMES_HOME/mongodb.json`` > defaults.

The schema returned by :func:`get_config_schema` is consumed directly by
``hermes memory setup`` to render an interactive form.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "mongodb.json"

DEFAULTS: dict[str, Any] = {
    "connection_uri": "",
    "database": "hermes_memory",
    "embedding_provider": "none",
    "embedding_api_key": "",
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536,
    "tenant_scope": "per-workspace",
    "auto_extract": False,
    "extraction_llm": False,
    "time_decay_half_life_days": 30,
    "default_ttl_days": 0,
    "hybrid_vector_weight": 0.6,
    "prefetch_limit": 5,
    "turn_ttl_days": 30,
}

# Env vars take precedence over the JSON file. We map each schema key to
# the env var Hermes documents elsewhere (URI, API keys), keeping the
# discovery surface predictable.
ENV_OVERRIDES: dict[str, str] = {
    "connection_uri": "HERMES_MONGODB_URI",
    "embedding_api_key": "HERMES_MONGODB_EMBEDDING_API_KEY",
}


def get_config_schema() -> list[dict[str, Any]]:
    """Return the field schema consumed by ``hermes memory setup``."""
    return [
        {
            "key": "connection_uri",
            "description": "MongoDB connection string (mongodb:// or mongodb+srv://)",
            "secret": True,
            "required": True,
            "env_var": "HERMES_MONGODB_URI",
            "url": "https://www.mongodb.com/cloud/atlas",
        },
        {
            "key": "database",
            "description": "Database name to use for memory collections",
            "default": DEFAULTS["database"],
        },
        {
            "key": "embedding_provider",
            "description": "Embedding provider for vector search (or 'none' for BM25-only)",
            "default": DEFAULTS["embedding_provider"],
            "choices": ["none", "openai", "voyage"],
        },
        {
            "key": "embedding_api_key",
            "description": "API key for the chosen embedding provider (skip if 'none')",
            "secret": True,
            "env_var": "HERMES_MONGODB_EMBEDDING_API_KEY",
        },
        {
            "key": "embedding_model",
            "description": "Embedding model name (e.g. text-embedding-3-small, voyage-3)",
            "default": DEFAULTS["embedding_model"],
        },
        {
            "key": "embedding_dim",
            "description": "Embedding dimension; must match your Atlas Vector Search index",
            "default": DEFAULTS["embedding_dim"],
        },
        {
            "key": "tenant_scope",
            "description": "Memory isolation scope",
            "default": DEFAULTS["tenant_scope"],
            "choices": ["global", "per-workspace", "per-profile"],
        },
        {
            "key": "auto_extract",
            "description": "Run regex extractor at session end to capture preferences/decisions",
            "default": str(DEFAULTS["auto_extract"]).lower(),
            "choices": ["true", "false"],
        },
        {
            "key": "extraction_llm",
            "description": "Use Hermes' LLM (when available) for richer session-end extraction",
            "default": str(DEFAULTS["extraction_llm"]).lower(),
            "choices": ["true", "false"],
        },
        {
            "key": "time_decay_half_life_days",
            "description": "Half-life (days) for time-decay scoring; 0 to disable",
            "default": DEFAULTS["time_decay_half_life_days"],
        },
        {
            "key": "default_ttl_days",
            "description": "Default TTL for new memories in days; 0 = never expire",
            "default": DEFAULTS["default_ttl_days"],
        },
        {
            "key": "hybrid_vector_weight",
            "description": "Vector vs BM25 weight (0.0-1.0) when both are available",
            "default": DEFAULTS["hybrid_vector_weight"],
        },
        {
            "key": "prefetch_limit",
            "description": "Number of memories injected into each turn's context",
            "default": DEFAULTS["prefetch_limit"],
        },
        {
            "key": "turn_ttl_days",
            "description": "TTL for raw turn capture (set 0 to keep forever)",
            "default": DEFAULTS["turn_ttl_days"],
        },
    ]


def _hermes_home() -> Path:
    """Resolve $HERMES_HOME with the documented default."""
    raw = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(raw).expanduser()


def _config_path(hermes_home: str | os.PathLike[str] | None = None) -> Path:
    base = Path(hermes_home).expanduser() if hermes_home else _hermes_home()
    return base / CONFIG_FILENAME


def load_config(hermes_home: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Merge defaults < file < env, returning the effective config."""
    cfg: dict[str, Any] = dict(DEFAULTS)

    path = _config_path(hermes_home)
    if path.is_file():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)

    for key, env_var in ENV_OVERRIDES.items():
        env_val = os.environ.get(env_var)
        if env_val:
            cfg[key] = env_val

    return cfg


def save_config(values: dict[str, Any], hermes_home: str | os.PathLike[str]) -> None:
    """Persist non-secret values to ``$HERMES_HOME/mongodb.json``.

    Secret fields (those backed by an env var) are intentionally not
    written to disk — Hermes' setup wizard stores them in ``.env``.
    """
    secret_keys = {f["key"] for f in get_config_schema() if f.get("secret")}
    payload = {k: v for k, v in values.items() if k not in secret_keys}

    path = _config_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def coerce_bool(value: Any) -> bool:
    """Coerce loosely-typed config booleans (``'true'``, ``1``, ``True``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False
