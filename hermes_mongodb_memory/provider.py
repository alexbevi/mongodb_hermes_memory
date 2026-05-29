"""MongoDB-backed implementation of the Hermes ``MemoryProvider`` ABC.

Filled in across subsequent commits. This module currently exposes a
minimal subclass so the plugin loader can discover the package.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_base() -> type:
    """Lazily resolve ``agent.memory_provider.MemoryProvider``.

    Hermes is not a runtime dependency of this plugin — we only need its
    base class when running inside Hermes. Tests substitute a stub.
    """
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
    """MongoDB-backed memory provider (scaffold; lifecycle filled in next)."""

    @property
    def name(self) -> str:
        return "mongodb"

    def is_available(self) -> bool:
        return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        return None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []
