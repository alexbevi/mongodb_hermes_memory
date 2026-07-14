"""Mongo-side persistence for memories, turns, profiles, and entities.

The :class:`MongoStore` owns the collections and indexes; it does *not*
own the embedding pipeline or search ranking — those live in their own
modules so unit tests can mock each layer in isolation.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure, PyMongoError

logger = logging.getLogger(__name__)

MEMORIES = "memories"
TURNS = "turns"
PROFILES = "profiles"
ENTITIES = "entities"

VALID_CATEGORIES = ("user_pref", "project", "fact", "decision", "general")

_ENTITY_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_entity(value: str) -> str:
    """Lower-case + collapse non-alphanumerics so 'Dark-Mode' == 'dark mode'."""
    return _ENTITY_NORMALIZE_RE.sub(" ", value.lower()).strip()


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclass(frozen=True)
class MemoryRecord:
    """Lightweight view of a memory document for callers that don't want raw dicts."""

    memory_id: str
    tenant_id: str
    category: str
    content: str
    entities: tuple[str, ...]
    tags: tuple[str, ...]
    trust: float
    created_at: dt.datetime
    score: float | None = None
    score_components: Mapping[str, float] | None = None

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> MemoryRecord:
        return cls(
            memory_id=str(doc["_id"]),
            tenant_id=doc.get("tenant_id", ""),
            category=doc.get("category", "general"),
            content=doc.get("content", ""),
            entities=tuple(doc.get("entities", ()) or ()),
            tags=tuple(doc.get("tags", ()) or ()),
            trust=float(doc.get("trust", 0.5)),
            created_at=doc.get("created_at", utcnow()),
            score=doc.get("score"),
            score_components=doc.get("score_components"),
        )


class MongoStore:
    """Wrap a :class:`pymongo.database.Database` with the plugin's data shape."""

    def __init__(self, db: Database, tenant_id: str = "default") -> None:
        self._db = db
        self._tenant_id = tenant_id

    @property
    def db(self) -> Database:
        return self._db

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def memories(self) -> Collection:
        return self._db[MEMORIES]

    @property
    def turns(self) -> Collection:
        return self._db[TURNS]

    @property
    def profiles(self) -> Collection:
        return self._db[PROFILES]

    @property
    def entities(self) -> Collection:
        return self._db[ENTITIES]

    # ------------------------------------------------------------------ indexes

    def ensure_indexes(self) -> None:
        """Create indexes idempotently. Safe to call on every initialize()."""
        try:
            self.memories.create_indexes(
                [
                    IndexModel(
                        [("tenant_id", ASCENDING), ("category", ASCENDING), ("created_at", DESCENDING)],
                        name="tenant_category_recent",
                    ),
                    IndexModel(
                        [("tenant_id", ASCENDING), ("entities", ASCENDING)],
                        name="tenant_entities",
                    ),
                    IndexModel(
                        [("tenant_id", ASCENDING), ("tags", ASCENDING)],
                        name="tenant_tags",
                        sparse=True,
                    ),
                    IndexModel(
                        [("expires_at", ASCENDING)],
                        name="ttl_expires_at",
                        expireAfterSeconds=0,
                        sparse=True,
                    ),
                    IndexModel([("content", "text")], name="content_text"),
                ]
            )
            self.turns.create_indexes(
                [
                    IndexModel(
                        [("tenant_id", ASCENDING), ("session_id", ASCENDING), ("turn_idx", ASCENDING)],
                        name="tenant_session_turn",
                    ),
                    IndexModel(
                        [("expires_at", ASCENDING)],
                        name="turns_ttl",
                        expireAfterSeconds=0,
                        sparse=True,
                    ),
                ]
            )
            self.entities.create_indexes(
                [
                    IndexModel(
                        [("tenant_id", ASCENDING), ("normalized", ASCENDING)],
                        name="tenant_entity_unique",
                        unique=True,
                    )
                ]
            )
        except OperationFailure as exc:
            # `text` indexes are unsupported in some emulators (mongomock without the
            # text-index extra). We log and continue — search.py degrades to in-process.
            logger.warning("Some indexes were not created: %s", exc)

    # ------------------------------------------------------------------ memories

    def add_memory(
        self,
        content: str,
        *,
        category: str = "general",
        entities: Iterable[str] = (),
        tags: Iterable[str] = (),
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
        ttl_days: int = 0,
        source: str = "tool",
        session_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Insert a memory; returns its id. Empty content is rejected."""
        if not content or not content.strip():
            raise ValueError("memory content must be non-empty")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category {category!r}; expected one of {VALID_CATEGORIES}")

        now = utcnow()
        normalized_entities = sorted({normalize_entity(e) for e in entities if e and e.strip()})
        doc: dict[str, Any] = {
            "_id": uuid.uuid4().hex,
            "tenant_id": self._tenant_id,
            "session_id": session_id,
            "category": category,
            "content": content.strip(),
            "entities": normalized_entities,
            "tags": sorted({t.strip() for t in tags if t and t.strip()}),
            "trust": 0.5,
            "helpful_count": 0,
            "unhelpful_count": 0,
            "source": source,
            "created_at": now,
            "updated_at": now,
            "metadata": dict(metadata or {}),
        }
        if embedding is not None:
            doc["embedding"] = list(embedding)
            doc["embedding_model"] = embedding_model or ""
        if ttl_days > 0:
            doc["expires_at"] = now + dt.timedelta(days=ttl_days)

        self.memories.insert_one(doc)
        self._upsert_entities(normalized_entities, now)
        return doc["_id"]

    def _upsert_entities(self, normalized: Iterable[str], when: dt.datetime) -> None:
        for ent in normalized:
            self.entities.update_one(
                {"tenant_id": self._tenant_id, "normalized": ent},
                {
                    "$setOnInsert": {
                        "tenant_id": self._tenant_id,
                        "normalized": ent,
                        "first_seen": when,
                    },
                    "$set": {"last_seen": when},
                    "$inc": {"mention_count": 1},
                },
                upsert=True,
            )

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        return self.memories.find_one({"_id": memory_id, "tenant_id": self._tenant_id})

    def remove_memory(self, memory_id: str) -> bool:
        res = self.memories.delete_one({"_id": memory_id, "tenant_id": self._tenant_id})
        return res.deleted_count > 0

    def expire_memory(self, memory_id: str, ttl_days: int) -> bool:
        if ttl_days <= 0:
            return self.remove_memory(memory_id)
        expires = utcnow() + dt.timedelta(days=ttl_days)
        res = self.memories.update_one(
            {"_id": memory_id, "tenant_id": self._tenant_id},
            {"$set": {"expires_at": expires, "updated_at": utcnow()}},
        )
        return res.matched_count > 0

    def list_memories(
        self,
        *,
        category: str | None = None,
        entities: Iterable[str] = (),
        min_trust: float = 0.0,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"tenant_id": self._tenant_id, "trust": {"$gte": min_trust}}
        if category:
            query["category"] = category
        ent_list = [normalize_entity(e) for e in entities if e and e.strip()]
        if ent_list:
            query["entities"] = {"$in": ent_list}
        cursor = self.memories.find(query).sort("created_at", DESCENDING).limit(int(limit))
        return list(cursor)

    def count_memories(self) -> int:
        try:
            return self.memories.count_documents({"tenant_id": self._tenant_id})
        except PyMongoError:
            return 0

    def record_feedback(self, memory_id: str, *, helpful: bool) -> dict[str, Any] | None:
        delta = 0.05 if helpful else -0.10
        field = "helpful_count" if helpful else "unhelpful_count"
        doc = self.memories.find_one({"_id": memory_id, "tenant_id": self._tenant_id})
        if not doc:
            return None
        new_trust = max(0.0, min(1.0, float(doc.get("trust", 0.5)) + delta))
        self.memories.update_one(
            {"_id": memory_id, "tenant_id": self._tenant_id},
            {"$set": {"trust": new_trust, "updated_at": utcnow()}, "$inc": {field: 1}},
        )
        return {"memory_id": memory_id, "trust": new_trust, "helpful": helpful}

    # ------------------------------------------------------------------ turns

    def append_turn(
        self,
        *,
        session_id: str,
        turn_idx: int,
        role: str,
        content: str,
        ttl_days: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not content:
            return
        now = utcnow()
        doc: dict[str, Any] = {
            "_id": uuid.uuid4().hex,
            "tenant_id": self._tenant_id,
            "session_id": session_id,
            "turn_idx": turn_idx,
            "role": role,
            "content": content,
            "created_at": now,
        }
        if metadata:
            doc["metadata"] = dict(metadata)
        if ttl_days > 0:
            doc["expires_at"] = now + dt.timedelta(days=ttl_days)
        self.turns.insert_one(doc)

    def next_turn_idx(self, session_id: str) -> int:
        latest = self.turns.find_one(
            {"tenant_id": self._tenant_id, "session_id": session_id},
            sort=[("turn_idx", DESCENDING)],
            projection={"turn_idx": 1},
        )
        return int(latest.get("turn_idx", 0) + 1) if latest else 1

    # ------------------------------------------------------------------ entities

    def related_memories(
        self,
        seed_entities: Iterable[str],
        *,
        max_depth: int = 2,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Walk the entity co-occurrence graph using $graphLookup when available.

        On engines that don't support $graphLookup (mongomock), falls back to
        a one-hop intersection on the ``tenant_entities`` index — which is
        exactly the same data shape, just without the recursive walk.
        """
        normalized = [normalize_entity(e) for e in seed_entities if e and e.strip()]
        if not normalized:
            return []

        try:
            pipeline: list[dict[str, Any]] = [
                {
                    "$match": {
                        "tenant_id": self._tenant_id,
                        "entities": {"$in": normalized},
                    }
                },
                {
                    "$graphLookup": {
                        "from": MEMORIES,
                        "startWith": "$entities",
                        "connectFromField": "entities",
                        "connectToField": "entities",
                        "as": "neighbors",
                        "maxDepth": max(0, max_depth - 1),
                        "depthField": "depth",
                        "restrictSearchWithMatch": {"tenant_id": self._tenant_id},
                    }
                },
                {"$project": {"_id": 0, "neighbors": 1}},
                {"$unwind": "$neighbors"},
                {"$replaceRoot": {"newRoot": "$neighbors"}},
                {"$group": {"_id": "$_id", "doc": {"$first": "$$ROOT"}}},
                {"$replaceRoot": {"newRoot": "$doc"}},
                {"$sort": {"depth": 1, "created_at": -1}},
                {"$limit": int(limit)},
            ]
            return list(self.memories.aggregate(pipeline))
        except (OperationFailure, NotImplementedError, PyMongoError) as exc:
            logger.debug("$graphLookup unsupported, falling back to one-hop: %s", exc)

        # one-hop fallback
        return list(
            self.memories.find(
                {"tenant_id": self._tenant_id, "entities": {"$in": normalized}}
            )
            .sort("created_at", DESCENDING)
            .limit(int(limit))
        )

    # ------------------------------------------------------------------ profiles

    def upsert_profile(self, summary: str, preferences: list[str]) -> None:
        self.profiles.update_one(
            {"_id": self._tenant_id},
            {
                "$set": {
                    "summary": summary,
                    "preferences": preferences,
                    "updated_at": utcnow(),
                }
            },
            upsert=True,
        )

    def get_profile(self) -> dict[str, Any] | None:
        return self.profiles.find_one({"_id": self._tenant_id})

    # ------------------------------------------------------------------ housekeeping

    def iter_recent(self, *, limit: int = 50) -> Iterator[dict[str, Any]]:
        yield from self.memories.find({"tenant_id": self._tenant_id}).sort("created_at", DESCENDING).limit(limit)

    def time_decay_weight(self, created_at: dt.datetime, half_life_days: float) -> float:
        if half_life_days <= 0:
            return 1.0
        # mongomock strips tzinfo; normalise both sides to be UTC-aware so the
        # subtraction always succeeds.
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=dt.timezone.utc)
        age_days = max(0.0, (utcnow() - created_at).total_seconds() / 86400.0)
        return math.exp(-math.log(2.0) * age_days / half_life_days)
