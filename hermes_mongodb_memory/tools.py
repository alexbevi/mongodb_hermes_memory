"""Tool schemas and dispatcher for the MongoDB memory provider.

Six tools are exposed:

* ``mongo_remember``  — store a memory with optional entities/tags/TTL
* ``mongo_search``    — hybrid query (vector + BM25 + time-decay)
* ``mongo_recall``    — entity-centric graph traversal
* ``mongo_forget``    — remove or schedule TTL on an existing memory
* ``mongo_profile``   — compact summary of preferences and project context
* ``mongo_reflect``   — cross-category synthesis grouped by entity

Each handler returns a JSON string envelope so Hermes can stream the
result back to the model unmodified.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .embeddings import EmbeddingClient, NullEmbeddingClient
from .search import HybridSearcher
from .store import VALID_CATEGORIES, MongoStore

logger = logging.getLogger(__name__)


def _category_enum_description() -> str:
    return f"One of: {', '.join(VALID_CATEGORIES)}"


def _tool_error(message: str) -> str:
    """Return Hermes-style error envelope; falls back to bare JSON if Hermes is absent."""
    try:
        from tools.registry import tool_error  # type: ignore[import-not-found]
        return tool_error(message)
    except Exception:
        return json.dumps({"error": message})


REMEMBER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mongo_remember",
        "description": (
            "Store a memory in MongoDB. Use for durable facts, user preferences, "
            "project decisions, or anything you want to recall in future sessions. "
            "Set ttl_days for ephemeral context that should self-expire."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory text to store"},
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": _category_enum_description(),
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Named entities the memory relates to (people, projects, tools)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Free-form tags for filtering",
                },
                "ttl_days": {
                    "type": "integer",
                    "description": "If > 0, memory auto-expires after this many days",
                },
            },
            "required": ["content"],
        },
    },
}

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mongo_search",
        "description": (
            "Hybrid search across MongoDB-backed memories: vector semantic + BM25 "
            "keyword, fused with reciprocal-rank fusion and time-decay scoring. "
            "ALWAYS call this before answering questions about the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query"},
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": "Optional category filter",
                },
                "entities": {"type": "array", "items": {"type": "string"}},
                "min_trust": {"type": "number", "description": "0.0-1.0 trust floor"},
                "limit": {"type": "integer", "description": "Top-K to return (default 5)"},
            },
            "required": ["query"],
        },
    },
}

RECALL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mongo_recall",
        "description": (
            "Walk the entity co-occurrence graph to find memories related to one "
            "or more named entities. Useful when the model has identified a "
            "subject and needs ALL related context, not just the closest match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Seed entities; the search expands via shared-entity links",
                },
                "depth": {"type": "integer", "description": "Graph traversal depth (default 2)"},
                "limit": {"type": "integer", "description": "Maximum results"},
            },
            "required": ["entities"],
        },
    },
}

FORGET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mongo_forget",
        "description": (
            "Remove a memory immediately, or schedule its expiry by setting "
            "ttl_days > 0. Use when the user retracts or supersedes information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "ttl_days": {
                    "type": "integer",
                    "description": "If > 0, schedule TTL; if 0 or omitted, delete immediately",
                },
            },
            "required": ["memory_id"],
        },
    },
}

PROFILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mongo_profile",
        "description": (
            "Summarize what we know about the user: latest preferences and the "
            "current project context. Cheap call; safe to use at conversation start."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

REFLECT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mongo_reflect",
        "description": (
            "Group recent memories by entity for the model to summarize themes "
            "and contradictions across categories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Memories to consider (default 50)"},
            },
        },
    },
}

ALL_SCHEMAS: list[dict[str, Any]] = [
    REMEMBER_SCHEMA,
    SEARCH_SCHEMA,
    RECALL_SCHEMA,
    FORGET_SCHEMA,
    PROFILE_SCHEMA,
    REFLECT_SCHEMA,
]


class ToolDispatcher:
    """Route tool calls from the provider to the underlying store/searcher."""

    def __init__(
        self,
        store: MongoStore,
        searcher: HybridSearcher,
        *,
        embedder: EmbeddingClient | None = None,
        default_ttl_days: int = 0,
    ) -> None:
        self._store = store
        self._searcher = searcher
        self._embedder = embedder or NullEmbeddingClient()
        self._default_ttl = int(default_ttl_days)

    def schemas(self) -> list[dict[str, Any]]:
        return list(ALL_SCHEMAS)

    def handle(self, name: str, args: Mapping[str, Any]) -> str:
        if not isinstance(args, Mapping):
            return _tool_error("tool args must be an object")
        try:
            handler = getattr(self, f"_handle_{name.replace('mongo_', '')}", None)
            if handler is None:
                return _tool_error(f"unknown tool: {name}")
            return handler(dict(args))
        except KeyError as exc:
            return _tool_error(f"missing required argument: {exc.args[0]}")
        except ValueError as exc:
            return _tool_error(str(exc))
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc, exc_info=True)
            return _tool_error(f"{name} failed: {exc}")

    # -------------------------------------------------------------- handlers

    def _handle_remember(self, args: dict[str, Any]) -> str:
        content = (args.get("content") or "").strip()
        if not content:
            return _tool_error("content must be a non-empty string")
        category = args.get("category", "general")
        if category not in VALID_CATEGORIES:
            return _tool_error(f"invalid category: {category}")
        ttl_days = int(args.get("ttl_days") or self._default_ttl)
        embedding: list[float] | None = None
        embedding_model: str | None = None
        if not isinstance(self._embedder, NullEmbeddingClient):
            try:
                embedding = self._embedder.embed(content)
                embedding_model = self._embedder.model
            except Exception as exc:
                logger.warning("embedding failed; storing without vector: %s", exc)

        memory_id = self._store.add_memory(
            content,
            category=category,
            entities=args.get("entities") or [],
            tags=args.get("tags") or [],
            embedding=embedding,
            embedding_model=embedding_model,
            ttl_days=ttl_days,
            source="tool",
        )
        return json.dumps(
            {
                "status": "stored",
                "memory_id": memory_id,
                "category": category,
                "ttl_days": ttl_days,
                "embedded": embedding is not None,
            }
        )

    def _handle_search(self, args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return _tool_error("query must be non-empty")
        limit = int(args.get("limit") or 5)
        hits = self._searcher.search(
            query,
            category=args.get("category"),
            entities=args.get("entities") or [],
            min_trust=float(args.get("min_trust") or 0.0),
            limit=limit,
        )
        return json.dumps({"results": [_serialise(h) for h in hits], "count": len(hits)})

    def _handle_recall(self, args: dict[str, Any]) -> str:
        entities = args.get("entities") or []
        if not entities:
            return _tool_error("entities must be a non-empty array")
        depth = int(args.get("depth") or 2)
        limit = int(args.get("limit") or 10)
        hits = self._store.related_memories(entities, max_depth=depth, limit=limit)
        return json.dumps({"results": [_serialise(h) for h in hits], "count": len(hits)})

    def _handle_forget(self, args: dict[str, Any]) -> str:
        memory_id = args["memory_id"]
        ttl_days = int(args.get("ttl_days") or 0)
        if ttl_days > 0:
            ok = self._store.expire_memory(memory_id, ttl_days)
            return json.dumps({"status": "ttl_set" if ok else "not_found", "memory_id": memory_id})
        ok = self._store.remove_memory(memory_id)
        return json.dumps({"status": "removed" if ok else "not_found", "memory_id": memory_id})

    def _handle_profile(self, args: dict[str, Any]) -> str:
        prefs = self._store.list_memories(category="user_pref", limit=10)
        projects = self._store.list_memories(category="project", limit=10)
        decisions = self._store.list_memories(category="decision", limit=5)
        return json.dumps(
            {
                "preferences": [_serialise(p) for p in prefs],
                "projects": [_serialise(p) for p in projects],
                "decisions": [_serialise(d) for d in decisions],
            }
        )

    def _handle_reflect(self, args: dict[str, Any]) -> str:
        limit = int(args.get("limit") or 50)
        recents = list(self._store.iter_recent(limit=limit))
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in recents:
            ents = doc.get("entities") or ["(unknown)"]
            for ent in ents:
                clusters[ent].append(_serialise(doc))
        # rank clusters by size, return top 8
        ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)[:8]
        return json.dumps({"clusters": [{"entity": k, "memories": v} for k, v in ordered]})


def _serialise(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Trim a raw mongo doc into something safe to hand to the model."""
    out: dict[str, Any] = {
        "memory_id": str(doc.get("_id", "")),
        "category": doc.get("category", "general"),
        "content": doc.get("content", ""),
        "entities": list(doc.get("entities") or []),
        "tags": list(doc.get("tags") or []),
        "trust": float(doc.get("trust", 0.5)),
    }
    if "score" in doc:
        out["score"] = float(doc["score"])
    if doc.get("score_components"):
        out["score_components"] = dict(doc["score_components"])
    if "created_at" in doc:
        ca = doc["created_at"]
        out["created_at"] = ca.isoformat() if hasattr(ca, "isoformat") else str(ca)
    return out
