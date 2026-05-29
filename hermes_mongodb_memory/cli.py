"""``hermes mongodb`` subcommands.

Hermes' plugin loader looks for ``register_cli(subparser)`` in a
sibling ``cli.py`` file when this provider is the active memory backend.
Three subcommands are exposed:

* ``init-indexes`` — create / refresh all collection + Atlas Search indexes.
* ``status``       — print connection, memory count, and atlas-detection state.
* ``doctor``       — best-effort connectivity + index audit, returns nonzero on issues.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from . import config as cfg
from ._atlas import looks_like_atlas_uri, server_supports_atlas_search

logger = logging.getLogger(__name__)


def register_cli(subparser: argparse._SubParsersAction) -> None:
    """Hermes calls this when our provider is active."""
    parser = subparser.add_parser("mongodb", help="MongoDB memory provider utilities")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("init-indexes", help="Create / refresh collection indexes")
    sub.add_parser("status", help="Show connection and memory counts")
    sub.add_parser("doctor", help="Run a connectivity + index audit")
    parser.set_defaults(func=mongodb_command)


def mongodb_command(args: argparse.Namespace) -> int:
    """Top-level entry point invoked by Hermes for the ``mongodb`` subcommand."""
    action = getattr(args, "action", None)
    if action == "init-indexes":
        return _cmd_init_indexes()
    if action == "status":
        return _cmd_status()
    if action == "doctor":
        return _cmd_doctor()
    print("usage: hermes mongodb {init-indexes,status,doctor}", file=sys.stderr)
    return 2


def _open_provider() -> tuple[Any, dict[str, Any]] | tuple[None, dict[str, Any]]:
    """Build a provider instance from the on-disk config; returns (provider, config)."""
    config = cfg.load_config()
    if not config.get("connection_uri"):
        print("error: connection_uri not set. Run `hermes memory setup` first.", file=sys.stderr)
        return None, config
    from .provider import MongoDBMemoryProvider

    provider = MongoDBMemoryProvider(config=config)
    if not provider.is_available():
        print("error: pymongo not installed or connection_uri empty.", file=sys.stderr)
        return None, config
    provider.initialize("cli")
    return provider, config


def _cmd_init_indexes() -> int:
    provider, _ = _open_provider()
    if provider is None:
        return 1
    try:
        provider._store.ensure_indexes()
        names = sorted(i["name"] for i in provider._store.memories.list_indexes())
        print("Created/verified indexes on `memories`:")
        for n in names:
            print(f"  - {n}")
        return 0
    finally:
        provider.shutdown()


def _cmd_status() -> int:
    provider, config = _open_provider()
    if provider is None:
        return 1
    try:
        client = provider._client
        atlas = looks_like_atlas_uri(config.get("connection_uri", "")) or server_supports_atlas_search(client)
        info = {
            "tenant_id": provider._tenant_id,
            "database": config.get("database"),
            "atlas_detected": bool(atlas),
            "embedding_provider": config.get("embedding_provider", "none"),
            "memories": provider._store.count_memories(),
        }
        print(json.dumps(info, indent=2))
        return 0
    finally:
        provider.shutdown()


def _cmd_doctor() -> int:
    """Lightweight connectivity + index audit."""
    provider, config = _open_provider()
    if provider is None:
        return 1
    issues: list[str] = []
    try:
        try:
            provider._client.admin.command("ping")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"ping failed: {exc}")

        memory_indexes = {i["name"] for i in provider._store.memories.list_indexes()}
        for required in ("tenant_category_recent", "tenant_entities", "ttl_expires_at"):
            if required not in memory_indexes:
                issues.append(f"missing index on memories: {required}")

        embedding_provider = (config.get("embedding_provider") or "none").lower()
        if embedding_provider != "none" and not config.get("embedding_api_key"):
            issues.append(
                f"embedding_provider={embedding_provider} but embedding_api_key is empty"
            )

        if issues:
            print("FAIL")
            for i in issues:
                print(f"  - {i}")
            return 1
        print("OK — all checks passed")
        return 0
    finally:
        provider.shutdown()


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="hermes mongodb")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init-indexes")
    sub.add_parser("status")
    sub.add_parser("doctor")
    args = parser.parse_args()
    sys.exit(mongodb_command(args))
