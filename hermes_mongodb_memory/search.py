"""Hybrid search across Mongo's vector + BM25 paths.

Two implementations live here:

* :meth:`HybridSearcher.search_atlas` — single ``$rankFusion`` aggregation
  blending Atlas Vector Search and Atlas Search BM25, with time-decay and
  trust factored in via ``$addFields``.
* :meth:`HybridSearcher.search_portable` — runs ``$text`` and an in-process
  cosine over stored embeddings, then merges results with reciprocal-rank
  fusion. Works on community/self-hosted Mongo and on the test mongomock.

The public :meth:`HybridSearcher.search` selects between them based on
``prefer_atlas`` (set at provider init) and degrades on
``OperationFailure``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Any

from pymongo.errors import OperationFailure, PyMongoError

from .embeddings import EmbeddingClient, NullEmbeddingClient
from .store import MEMORIES, MongoStore, normalize_entity, utcnow

logger = logging.getLogger(__name__)

ATLAS_TEXT_INDEX = "hermes_memory_text"
ATLAS_VECTOR_INDEX = "hermes_memory_vector"

# Reciprocal-rank fusion constant — k=60 is the standard from the
# Atlas $rankFusion documentation, and works well in the portable path
# too.
_RRF_K = 60


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


class HybridSearcher:
    def __init__(
        self,
        store: MongoStore,
        embedder: EmbeddingClient,
        *,
        prefer_atlas: bool = False,
        time_decay_half_life_days: float = 30.0,
        vector_weight: float = 0.6,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._prefer_atlas = prefer_atlas
        self._half_life = float(time_decay_half_life_days)
        self._vector_weight = float(vector_weight)

    @property
    def embedder(self) -> EmbeddingClient:
        return self._embedder

    # -------------------------------------------------------------- public

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        entities: Iterable[str] = (),
        min_trust: float = 0.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        ent_filter = sorted({normalize_entity(e) for e in entities if e})

        if self._prefer_atlas:
            try:
                hits = self.search_atlas(
                    query, category=category, entities=ent_filter, min_trust=min_trust, limit=limit
                )
                if hits:
                    return hits
            except (OperationFailure, PyMongoError, NotImplementedError) as exc:
                logger.warning("Atlas $rankFusion path failed, falling back: %s", exc)
        return self.search_portable(
            query, category=category, entities=ent_filter, min_trust=min_trust, limit=limit
        )

    # -------------------------------------------------------------- atlas

    def search_atlas(
        self,
        query: str,
        *,
        category: str | None,
        entities: Sequence[str],
        min_trust: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """$rankFusion over $vectorSearch + $search; requires Atlas 8.1+."""
        match_stage: dict[str, Any] = {"tenant_id": self._store.tenant_id}
        if category:
            match_stage["category"] = category
        if entities:
            match_stage["entities"] = {"$in": list(entities)}
        if min_trust > 0:
            match_stage["trust"] = {"$gte": float(min_trust)}

        text_pipeline: list[dict[str, Any]] = [
            {
                "$search": {
                    "index": ATLAS_TEXT_INDEX,
                    "text": {"query": query, "path": "content"},
                }
            },
            {"$match": match_stage},
            {"$limit": int(limit) * 5},
        ]
        pipelines: dict[str, list[dict[str, Any]]] = {"text": text_pipeline}

        if not isinstance(self._embedder, NullEmbeddingClient):
            try:
                vec = self._embedder.embed(query)
            except Exception as exc:
                logger.warning("embedding for query failed: %s", exc)
                vec = []
            if vec:
                pipelines["vector"] = [
                    {
                        "$vectorSearch": {
                            "index": ATLAS_VECTOR_INDEX,
                            "path": "embedding",
                            "queryVector": vec,
                            "numCandidates": 200,
                            "limit": int(limit) * 5,
                            "filter": match_stage,
                        }
                    }
                ]

        weights = self._fusion_weights(pipelines)
        rank_fusion: dict[str, Any] = {
            "$rankFusion": {
                "input": {"pipelines": pipelines},
                "combination": {"weights": weights},
            }
        }

        pipeline: list[dict[str, Any]] = [
            rank_fusion,
            self._decay_stage(),
            {"$sort": {"score": -1}},
            {"$limit": int(limit)},
        ]
        return list(self._store.memories.aggregate(pipeline))

    def _fusion_weights(self, pipelines: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
        if "vector" in pipelines:
            return {
                "vector": self._vector_weight,
                "text": max(0.0, 1.0 - self._vector_weight),
            }
        return {"text": 1.0}

    def _decay_stage(self) -> dict[str, Any]:
        if self._half_life <= 0:
            return {"$addFields": {"score": {"$ifNull": ["$score", 1.0]}}}
        # weight = exp(-ln(2) * age_days / half_life)
        return {
            "$addFields": {
                "score": {
                    "$multiply": [
                        {"$ifNull": ["$score", 1.0]},
                        {
                            "$exp": {
                                "$multiply": [
                                    -math.log(2.0) / self._half_life,
                                    {
                                        "$divide": [
                                            {"$subtract": [utcnow(), "$created_at"]},
                                            1000 * 60 * 60 * 24,
                                        ]
                                    },
                                ]
                            }
                        },
                    ]
                }
            }
        }

    # -------------------------------------------------------------- portable

    def search_portable(
        self,
        query: str,
        *,
        category: str | None,
        entities: Sequence[str],
        min_trust: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """BM25 (or keyword fallback) plus optional in-process cosine, RRF-merged."""
        text_hits = self._text_search(query, category=category, entities=entities, min_trust=min_trust, limit=limit * 5)

        vector_hits: list[dict[str, Any]] = []
        if not isinstance(self._embedder, NullEmbeddingClient):
            try:
                qvec = self._embedder.embed(query)
            except Exception as exc:
                logger.warning("embedding query failed: %s", exc)
                qvec = []
            if qvec:
                vector_hits = self._cosine_rerank(qvec, category=category, entities=entities, min_trust=min_trust, limit=limit * 5)

        merged = self._reciprocal_rank_fusion(text_hits, vector_hits)
        scored = [self._apply_decay(doc) for doc in merged]
        scored.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return scored[: int(limit)]

    def _text_search(
        self,
        query: str,
        *,
        category: str | None,
        entities: Sequence[str],
        min_trust: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        match: dict[str, Any] = {"tenant_id": self._store.tenant_id}
        if category:
            match["category"] = category
        if entities:
            match["entities"] = {"$in": list(entities)}
        if min_trust > 0:
            match["trust"] = {"$gte": float(min_trust)}

        try:
            cursor = (
                self._store.memories.find(
                    {"$text": {"$search": query}, **match},
                    {"score": {"$meta": "textScore"}, "_id": 1, "tenant_id": 1, "category": 1,
                     "content": 1, "entities": 1, "tags": 1, "trust": 1, "created_at": 1, "embedding": 1},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(int(limit))
            )
            return list(cursor)
        except (OperationFailure, PyMongoError, TypeError, NotImplementedError) as exc:
            # mongomock raises TypeError for $meta sort and NotImplementedError for some
            # operators — both signal that we should use the substring fallback.
            logger.debug("$text unavailable, falling back to substring filter: %s", exc)

        # mongomock / engines without text indexes: case-insensitive substring scan,
        # ranked by occurrence count.
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []
        all_docs = list(
            self._store.memories.find(match)
            .sort("created_at", -1)
            .limit(int(limit) * 5)
        )
        scored: list[dict[str, Any]] = []
        for doc in all_docs:
            content = (doc.get("content") or "").lower()
            score = sum(content.count(t) for t in terms)
            if score > 0:
                doc = dict(doc)
                doc["score"] = float(score)
                scored.append(doc)
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[: int(limit)]

    def _cosine_rerank(
        self,
        qvec: Sequence[float],
        *,
        category: str | None,
        entities: Sequence[str],
        min_trust: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        match: dict[str, Any] = {"tenant_id": self._store.tenant_id, "embedding": {"$exists": True}}
        if category:
            match["category"] = category
        if entities:
            match["entities"] = {"$in": list(entities)}
        if min_trust > 0:
            match["trust"] = {"$gte": float(min_trust)}

        candidates = list(self._store.memories.find(match).limit(500))
        scored = []
        for doc in candidates:
            sim = _cosine(qvec, doc.get("embedding") or [])
            if sim > 0:
                d = dict(doc)
                d["score"] = sim
                scored.append(d)
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[: int(limit)]

    def _reciprocal_rank_fusion(
        self,
        text_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        weights = {"text": 1.0, "vector": 0.0}
        if vector_hits:
            weights = {"text": max(0.0, 1.0 - self._vector_weight), "vector": self._vector_weight}

        scores: dict[Any, float] = {}
        components: dict[Any, dict[str, float]] = {}
        docs: dict[Any, dict[str, Any]] = {}

        for source, hits in (("text", text_hits), ("vector", vector_hits)):
            for rank, doc in enumerate(hits, start=1):
                key = doc["_id"]
                contribution = weights[source] / (_RRF_K + rank)
                scores[key] = scores.get(key, 0.0) + contribution
                components.setdefault(key, {})[source] = float(doc.get("score", 0.0))
                docs.setdefault(key, doc)

        merged = []
        for key, score in scores.items():
            doc = dict(docs[key])
            doc["score"] = score
            doc["score_components"] = components[key]
            merged.append(doc)
        return merged

    def _apply_decay(self, doc: dict[str, Any]) -> dict[str, Any]:
        if self._half_life <= 0:
            return doc
        created = doc.get("created_at")
        if created is None:
            return doc
        weight = self._store.time_decay_weight(created if hasattr(created, "tzinfo") else created, self._half_life)
        doc["score"] = float(doc.get("score", 0.0)) * weight
        return doc
