"""MongoDB memory provider plugin for Hermes Agent.

The Hermes plugin loader scans this file for the strings
``register_memory_provider`` and ``MemoryProvider`` to identify it as a
memory plugin, then calls ``register(ctx)``.
"""

from __future__ import annotations

from .provider import MongoDBMemoryProvider

__all__ = ["MongoDBMemoryProvider", "register"]
__version__ = "0.1.0"


def register(ctx) -> None:
    """Register the MongoDB memory provider with the plugin system."""
    ctx.register_memory_provider(MongoDBMemoryProvider())
