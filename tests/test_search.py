"""Tests for HybridSearcher (portable path; Atlas path covered in integration)."""

from __future__ import annotations

import datetime as dt

import pytest

from hermes_mongodb_memory.embeddings import NullEmbeddingClient
from hermes_mongodb_memory.search import HybridSearcher, _cosine
from hermes_mongodb_memory.store import utcnow


class StubEmbedder:
    name = "stub"
    model = "stub"
    dim = 3

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._map = mapping

    def embed(self, text: str) -> list[float]:
        return self._map.get(text, [0.0, 0.0, 0.0])

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine([], [1, 1]) == 0.0


def test_search_returns_empty_for_empty_query(store):
    s = HybridSearcher(store, NullEmbeddingClient())
    assert s.search("") == []
    assert s.search("   ") == []


def test_portable_text_search_finds_matching_memory(store):
    store.add_memory("I prefer dark mode", category="user_pref")
    store.add_memory("I drink coffee daily", category="fact")
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    hits = s.search("dark mode", limit=5)
    assert hits
    assert hits[0]["content"] == "I prefer dark mode"


def test_portable_search_filters_by_category(store):
    store.add_memory("I prefer dark mode", category="user_pref")
    store.add_memory("dark mode mention in project", category="project")
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    hits = s.search("dark mode", category="user_pref", limit=5)
    assert len(hits) == 1
    assert hits[0]["category"] == "user_pref"


def test_portable_search_filters_by_entities(store):
    store.add_memory("uses dark mode", category="fact", entities=["theme"])
    store.add_memory("loves the dark mode", category="fact", entities=["mood"])
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    hits = s.search("dark mode", entities=["theme"], limit=5)
    assert len(hits) == 1
    assert "theme" in hits[0]["entities"]


def test_portable_search_filters_by_min_trust(store):
    mid_high = store.add_memory("very trusted", category="fact")
    store.add_memory("untrusted", category="fact")
    # bump one's trust
    for _ in range(10):
        store.record_feedback(mid_high, helpful=True)
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    hits = s.search("trusted", min_trust=0.7, limit=5)
    assert len(hits) == 1
    assert hits[0]["_id"] == mid_high


def test_portable_search_returns_score_components_when_vector_used(store):
    embedder = StubEmbedder(
        {
            "dark mode": [1.0, 0.0, 0.0],
            "I prefer dark mode": [0.95, 0.05, 0.0],
            "I drink coffee": [0.0, 1.0, 0.0],
        }
    )
    store.add_memory(
        "I prefer dark mode",
        category="user_pref",
        embedding=embedder.embed("I prefer dark mode"),
    )
    store.add_memory(
        "I drink coffee", category="fact", embedding=embedder.embed("I drink coffee")
    )
    s = HybridSearcher(store, embedder, time_decay_half_life_days=0)
    hits = s.search("dark mode", limit=5)
    assert hits
    top = hits[0]
    assert top["content"] == "I prefer dark mode"
    assert "score_components" in top
    assert "vector" in top["score_components"]


def test_time_decay_drops_older_memories(store):
    new_id = store.add_memory("recent dark mode insight", category="fact")
    # Backdate one memory
    store.add_memory("ancient dark mode lore", category="fact")
    old_doc = store.memories.find_one({"content": "ancient dark mode lore"})
    store.memories.update_one(
        {"_id": old_doc["_id"]},
        {"$set": {"created_at": utcnow() - dt.timedelta(days=365)}},
    )

    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=30)
    hits = s.search("dark mode", limit=5)
    assert hits[0]["_id"] == new_id


def test_atlas_path_skipped_when_prefer_atlas_false(store, monkeypatch):
    """search() must not call search_atlas when prefer_atlas=False."""
    s = HybridSearcher(store, NullEmbeddingClient(), prefer_atlas=False)

    def boom(*args, **kwargs):
        raise AssertionError("atlas path should not run")

    monkeypatch.setattr(s, "search_atlas", boom)
    s.search("hello")  # should not raise


def test_atlas_path_failure_falls_back(store):
    s = HybridSearcher(store, NullEmbeddingClient(), prefer_atlas=True, time_decay_half_life_days=0)

    def fail(*args, **kwargs):
        from pymongo.errors import OperationFailure

        raise OperationFailure("no atlas search index")

    s.search_atlas = fail  # type: ignore[assignment]
    store.add_memory("dark mode hello", category="fact")
    hits = s.search("dark mode")
    assert hits  # portable path returned a result


def test_search_returns_empty_when_nothing_matches(store):
    store.add_memory("apples and oranges", category="fact")
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    assert s.search("zzzz nothing") == []


def test_build_atlas_pipeline_text_only_when_no_embedder(store):
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    pipeline = s.build_atlas_pipeline("dark mode", limit=5)
    rank_fusion = pipeline[0]["$rankFusion"]
    assert set(rank_fusion["input"]["pipelines"].keys()) == {"text"}
    assert rank_fusion["combination"]["weights"] == {"text": 1.0}
    # Final stages: decay -> sort -> limit
    assert pipeline[-2] == {"$sort": {"score": -1}}
    assert pipeline[-1] == {"$limit": 5}


def test_build_atlas_pipeline_includes_vector_when_embedding_present(store):
    embedder = StubEmbedder({"dark mode": [1.0, 0.0, 0.0]})
    s = HybridSearcher(store, embedder, time_decay_half_life_days=0, vector_weight=0.7)
    pipeline = s.build_atlas_pipeline("dark mode", limit=5)
    inputs = pipeline[0]["$rankFusion"]["input"]["pipelines"]
    assert set(inputs.keys()) == {"text", "vector"}
    weights = pipeline[0]["$rankFusion"]["combination"]["weights"]
    assert weights["vector"] == pytest.approx(0.7)
    assert weights["text"] == pytest.approx(0.3)
    vector_stage = inputs["vector"][0]["$vectorSearch"]
    assert vector_stage["queryVector"] == [1.0, 0.0, 0.0]
    assert vector_stage["index"] == "hermes_memory_vector"
    assert vector_stage["filter"]["tenant_id"] == "t1"


def test_build_atlas_pipeline_applies_filters(store):
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    pipeline = s.build_atlas_pipeline(
        "topic", category="user_pref", entities=["theme"], min_trust=0.4, limit=3
    )
    text_match = pipeline[0]["$rankFusion"]["input"]["pipelines"]["text"][1]["$match"]
    assert text_match["category"] == "user_pref"
    assert text_match["entities"] == {"$in": ["theme"]}
    assert text_match["trust"] == {"$gte": 0.4}


def test_build_atlas_pipeline_includes_decay_stage_when_half_life_set(store):
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=30)
    pipeline = s.build_atlas_pipeline("topic")
    decay = pipeline[1]["$addFields"]["score"]
    # When half-life > 0 we apply $multiply (not the no-op $ifNull-only stage).
    assert "$multiply" in decay


def test_build_atlas_pipeline_skips_decay_when_disabled(store):
    s = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    pipeline = s.build_atlas_pipeline("topic")
    decay = pipeline[1]["$addFields"]["score"]
    assert "$multiply" not in decay
    assert "$ifNull" in decay


def test_build_atlas_pipeline_skips_vector_when_embed_fails(store):
    class BoomEmbedder:
        name = "boom"
        model = "boom"
        dim = 3

        def embed(self, text):
            raise RuntimeError("upstream down")

        def embed_many(self, texts):
            return [[] for _ in texts]

    s = HybridSearcher(store, BoomEmbedder(), time_decay_half_life_days=0)
    pipeline = s.build_atlas_pipeline("topic")
    inputs = pipeline[0]["$rankFusion"]["input"]["pipelines"]
    # Embedding failure must drop the vector pipeline cleanly, leaving text intact.
    assert set(inputs.keys()) == {"text"}


def test_search_atlas_executes_built_pipeline(store, monkeypatch):
    """search_atlas must hand build_atlas_pipeline's output to aggregate()."""
    s = HybridSearcher(store, NullEmbeddingClient(), prefer_atlas=True, time_decay_half_life_days=0)

    captured = {}

    def fake_aggregate(pipeline):
        captured["pipeline"] = list(pipeline)
        return iter([{"_id": "x", "content": "stub", "score": 1.0}])

    monkeypatch.setattr(store.memories, "aggregate", fake_aggregate)
    out = s.search_atlas("hello", category=None, entities=[], min_trust=0.0, limit=5)
    assert out == [{"_id": "x", "content": "stub", "score": 1.0}]
    assert "$rankFusion" in captured["pipeline"][0]


def test_rrf_merge_combines_text_and_vector_results(store):
    embedder = StubEmbedder(
        {
            "topic": [1.0, 0.0, 0.0],
            "vector-only-content": [1.0, 0.0, 0.0],
            "text-only-topic": [0.0, 1.0, 0.0],
        }
    )
    store.add_memory(
        "vector-only-content",
        category="fact",
        embedding=embedder.embed("vector-only-content"),
    )
    store.add_memory(
        "text-only-topic", category="fact", embedding=embedder.embed("text-only-topic")
    )
    s = HybridSearcher(store, embedder, time_decay_half_life_days=0, vector_weight=0.5)
    hits = s.search("topic", limit=5)
    contents = {h["content"] for h in hits}
    assert "vector-only-content" in contents
    assert "text-only-topic" in contents
