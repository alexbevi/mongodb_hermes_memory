"""MongoDB-backed implementation of the Hermes ``MemoryProvider`` ABC.

The provider stitches together :mod:`config`, :mod:`store`,
:mod:`embeddings`, :mod:`search`, :mod:`tools`, and :mod:`extraction`.
It is deliberately the only module in the plugin that touches both the
network and Hermes' lifecycle hooks, so unit tests for the rest stay
fast and pure.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

from . import config as cfg
from ._atlas import looks_like_atlas_uri, server_supports_atlas_search
from .embeddings import EmbeddingClient, NullEmbeddingClient, make_embedding_client
from .extraction import LLMExtractor, RegexExtractor
from .search import HybridSearcher
from .store import MongoStore, normalize_entity
from .tools import ToolDispatcher

logger = logging.getLogger(__name__)


def _resolve_driver_info() -> Any:
    """Build a PyMongo ``DriverInfo`` for handshake attribution.

    Surfaces this plugin in MongoDB's server-side telemetry so deployment
    operators can see traffic from ``Hermes-MongoDB-Memory`` distinctly
    from generic PyMongo or other wrappers. Returns ``None`` if either
    PyMongo's DriverInfo or the package version isn't importable, so a
    handshake annotation never blocks the connection.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        from pymongo.driver_info import DriverInfo
    except ImportError:
        return None

    try:
        pkg_version = version("hermes-mongodb-memory")
    except PackageNotFoundError:
        pkg_version = None

    return DriverInfo(name="Hermes-MongoDB-Memory", version=pkg_version)


_DRIVER_INFO = _resolve_driver_info()


# Circuit breaker tuning: matches the mem0 plugin so behaviour is consistent
# across the ecosystem.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECONDS = 120


def _resolve_base() -> type:
    try:
        from agent.memory_provider import MemoryProvider  # type: ignore[import-not-found]
        return MemoryProvider
    except Exception:  # pragma: no cover - exercised only without Hermes installed
        from abc import ABC, abstractmethod

        class _StubMemoryProvider(ABC):
            @property
            @abstractmethod
            def name(self) -> str: ...

            @abstractmethod
            def is_available(self) -> bool: ...

            @abstractmethod
            def initialize(self, session_id: str, **kwargs: Any) -> None: ...

            @abstractmethod
            def get_tool_schemas(self) -> list[dict[str, Any]]: ...

        return _StubMemoryProvider


_MemoryProvider = _resolve_base()


class MongoDBMemoryProvider(_MemoryProvider):  # type: ignore[misc, valid-type]
    """MongoDB memory provider — hybrid search, TTL forgetting, entity graph."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = dict(config or cfg.load_config())
        self._client: Any = None
        self._db: Any = None
        self._store: MongoStore | None = None
        self._embedder: EmbeddingClient = NullEmbeddingClient()
        self._searcher: HybridSearcher | None = None
        self._dispatcher: ToolDispatcher | None = None

        self._session_id: str = ""
        self._tenant_id: str = "default"
        self._workspace: str = ""
        self._identity: str = ""

        # Background workers + circuit breaker
        self._sync_lock = threading.Lock()
        self._sync_thread: threading.Thread | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_result: str = ""
        self._failures = 0
        self._breaker_opened_at = 0.0

    # ------------------------------------------------------------------ identity

    @property
    def name(self) -> str:
        return "mongodb"

    def is_available(self) -> bool:
        try:
            import pymongo  # noqa: F401
        except ImportError:
            return False
        uri = self._config.get("connection_uri") or ""
        return bool(uri.strip())

    # ------------------------------------------------------------------ config / setup

    def get_config_schema(self) -> list[dict[str, Any]]:
        return cfg.get_config_schema()

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        cfg.save_config(values, hermes_home)

    def post_setup(self, hermes_home: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Verify connectivity and create indexes after the wizard finishes."""
        merged = dict(cfg.load_config(hermes_home))
        if config:
            merged.update(config)
        self._config = merged

        if not merged.get("connection_uri"):
            return {"ok": False, "reason": "connection_uri not set"}

        try:
            self._open_client(merged)
            assert self._store is not None
            self._store.ensure_indexes()
            return {
                "ok": True,
                "atlas": self._is_atlas,
                "tenant_id": self._tenant_id,
                "memories": self._store.count_memories(),
            }
        except Exception as exc:
            logger.warning("post_setup connectivity check failed: %s", exc)
            return {"ok": False, "reason": str(exc)}
        finally:
            self.shutdown()

    # ------------------------------------------------------------------ lifecycle

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._workspace = str(kwargs.get("agent_workspace") or "")
        self._identity = str(kwargs.get("agent_identity") or "")
        self._tenant_id = self._derive_tenant()

        self._open_client(self._config)
        assert self._store is not None
        self._store.ensure_indexes()

    def _derive_tenant(self) -> str:
        scope = self._config.get("tenant_scope", "per-workspace")
        if scope == "global":
            return "default"
        if scope == "per-profile" and self._identity:
            return f"profile:{self._identity}"
        if self._workspace:
            return f"ws:{normalize_entity(self._workspace)}"
        return "default"

    def _open_client(self, config: Mapping[str, Any]) -> None:
        from pymongo import MongoClient  # imported here so tests can patch

        uri = config["connection_uri"]
        client_kwargs: dict[str, Any] = {
            "serverSelectionTimeoutMS": int(config.get("server_selection_timeout_ms", 3000)),
        }
        if _DRIVER_INFO is not None:
            client_kwargs["driver"] = _DRIVER_INFO
        client = MongoClient(uri, **client_kwargs)
        self._client = client
        self._db = client[config.get("database", "hermes_memory")]
        self._store = MongoStore(self._db, tenant_id=self._tenant_id)
        self._embedder = make_embedding_client(dict(config))
        self._is_atlas = looks_like_atlas_uri(uri) or server_supports_atlas_search(client)
        self._searcher = HybridSearcher(
            self._store,
            self._embedder,
            prefer_atlas=self._is_atlas,
            time_decay_half_life_days=float(config.get("time_decay_half_life_days", 30)),
            vector_weight=float(config.get("hybrid_vector_weight", 0.6)),
        )
        self._dispatcher = ToolDispatcher(
            self._store,
            self._searcher,
            embedder=self._embedder,
            default_ttl_days=int(config.get("default_ttl_days", 0)),
        )

    def shutdown(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception as exc:
            logger.debug("MongoDB client close failed: %s", exc)
        self._client = None
        self._db = None
        self._store = None
        self._searcher = None
        self._dispatcher = None
        self._embedder = NullEmbeddingClient()

    # ------------------------------------------------------------------ system prompt

    def system_prompt_block(self) -> str:
        if self._store is None:
            return ""
        try:
            count = self._store.count_memories()
        except Exception:
            count = 0
        if count == 0:
            return (
                "# MongoDB Memory\n"
                "Active. No memories yet. Use `mongo_remember` to capture preferences,\n"
                "decisions, and durable facts so they're available next session."
            )
        return (
            "# MongoDB Memory\n"
            f"Active. {count} memories stored. Use `mongo_search` when prior "
            "preferences, decisions, facts, or project history may matter; use "
            "`mongo_profile` for an overview."
        )

    # ------------------------------------------------------------------ tools

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return self._dispatcher.schemas() if self._dispatcher else []

    def handle_tool_call(self, tool_name: str, args: Mapping[str, Any], **kwargs: Any) -> str:
        if self._dispatcher is None:
            return '{"error": "MongoDB memory provider is not initialized"}'
        return self._dispatcher.handle(tool_name, args)

    # ------------------------------------------------------------------ prefetch

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if result:
            return result
        return self._do_prefetch(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open() or self._searcher is None:
            return
        thread = threading.Thread(
            target=self._fill_prefetch, args=(query,), daemon=True, name="mongo-prefetch"
        )
        thread.start()
        self._prefetch_thread = thread

    def _fill_prefetch(self, query: str) -> None:
        try:
            text = self._do_prefetch(query)
            with self._prefetch_lock:
                self._prefetch_result = text
            self._record_success()
        except Exception as exc:
            self._record_failure()
            logger.debug("prefetch failed: %s", exc)

    def _do_prefetch(self, query: str) -> str:
        if not query or self._searcher is None:
            return ""
        limit = int(self._config.get("prefetch_limit", 5))
        hits = self._searcher.search(query, limit=limit)
        if not hits:
            return ""
        lines = ["## MongoDB Memory"]
        for h in hits:
            content = (h.get("content") or "").replace("\n", " ").strip()
            trust = float(h.get("trust", 0.5))
            lines.append(f"- [{trust:.2f}] {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ sync

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._is_breaker_open() or self._store is None:
            return

        sid = session_id or self._session_id
        ttl_days = int(self._config.get("turn_ttl_days", 30))

        def _do_sync() -> None:
            try:
                with self._sync_lock:
                    idx = self._store.next_turn_idx(sid)
                    if user_content:
                        self._store.append_turn(
                            session_id=sid, turn_idx=idx, role="user",
                            content=user_content, ttl_days=ttl_days,
                        )
                    if assistant_content:
                        self._store.append_turn(
                            session_id=sid, turn_idx=idx + 1, role="assistant",
                            content=assistant_content, ttl_days=ttl_days,
                        )
                self._record_success()
            except Exception as exc:
                self._record_failure()
                logger.warning("MongoDB sync_turn failed: %s", exc)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        thread = threading.Thread(target=_do_sync, daemon=True, name="mongo-sync")
        thread.start()
        self._sync_thread = thread

    # ------------------------------------------------------------------ session-end

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if self._store is None or not messages:
            return
        if not cfg.coerce_bool(self._config.get("auto_extract")):
            return

        regex_out = RegexExtractor().extract(messages)
        llm_out = []
        if cfg.coerce_bool(self._config.get("extraction_llm")):
            llm_out = LLMExtractor().extract(messages)

        for mem in regex_out + llm_out:
            try:
                self._store.add_memory(
                    mem.content,
                    category=mem.category,
                    entities=mem.entities,
                    tags=mem.tags,
                    source="auto_extract",
                    session_id=self._session_id,
                )
            except Exception as exc:
                logger.debug("auto-extract add_memory failed: %s", exc)

    # ------------------------------------------------------------------ session-switch

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs: Any,
    ) -> None:
        with self._prefetch_lock:
            self._prefetch_result = ""
        self._session_id = new_session_id

    # ------------------------------------------------------------------ memory-mirror

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._store is None or action != "add" or not content:
            return
        category = "user_pref" if target == "user" else "general"
        try:
            self._store.add_memory(
                content,
                category=category,
                source=f"mirror:{target}",
                session_id=self._session_id,
                metadata=dict(metadata or {}),
            )
        except Exception as exc:
            logger.debug("mirror add_memory failed: %s", exc)

    # ------------------------------------------------------------------ pre-compress

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        if self._store is None:
            return ""
        try:
            recent = self._store.list_memories(category="user_pref", limit=5)
        except Exception:
            return ""
        if not recent:
            return ""
        lines = ["High-trust memories (preserve through compression):"]
        for doc in recent:
            lines.append(f"- {doc.get('content', '')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ breaker

    def _is_breaker_open(self) -> bool:
        if self._failures < _BREAKER_THRESHOLD:
            return False
        return (time.time() - self._breaker_opened_at) < _BREAKER_COOLDOWN_SECONDS

    def _record_success(self) -> None:
        self._failures = 0
        self._breaker_opened_at = 0.0

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= _BREAKER_THRESHOLD:
            self._breaker_opened_at = time.time()
