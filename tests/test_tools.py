"""Tests for the tool dispatcher."""

from __future__ import annotations

import json

import pytest

from hermes_mongodb_memory.embeddings import NullEmbeddingClient
from hermes_mongodb_memory.search import HybridSearcher
from hermes_mongodb_memory.tools import ALL_SCHEMAS, ToolDispatcher


@pytest.fixture
def dispatcher(store):
    searcher = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    return ToolDispatcher(store, searcher)


def test_schemas_lists_all_tools(dispatcher):
    names = {s["function"]["name"] for s in dispatcher.schemas()}
    assert names == {
        "mongo_remember",
        "mongo_search",
        "mongo_recall",
        "mongo_forget",
        "mongo_profile",
        "mongo_reflect",
    }


def test_schemas_module_constant_matches():
    names = {s["function"]["name"] for s in ALL_SCHEMAS}
    assert "mongo_remember" in names
    assert "mongo_reflect" in names


def test_search_schema_uses_targeted_recall_guidance():
    schema = next(s for s in ALL_SCHEMAS if s["function"]["name"] == "mongo_search")
    description = schema["function"]["description"]
    assert "prior preferences" in description
    assert "self-contained requests" in description
    assert "ALWAYS call" not in description


def test_remember_stores_memory(dispatcher, store):
    res = json.loads(
        dispatcher.handle(
            "mongo_remember",
            {"content": "I prefer dark mode", "category": "user_pref", "entities": ["theme"]},
        )
    )
    assert res["status"] == "stored"
    assert res["memory_id"]
    assert store.count_memories() == 1
    doc = store.memories.find_one({"_id": res["memory_id"]})
    assert doc["content"] == "I prefer dark mode"
    assert "theme" in doc["entities"]


def test_remember_rejects_empty_content(dispatcher):
    res = json.loads(dispatcher.handle("mongo_remember", {"content": "  "}))
    assert "error" in res


def test_remember_rejects_invalid_category(dispatcher):
    res = json.loads(
        dispatcher.handle("mongo_remember", {"content": "x", "category": "invalid"})
    )
    assert "error" in res


def test_remember_with_ttl_sets_expires(dispatcher, store):
    res = json.loads(
        dispatcher.handle(
            "mongo_remember",
            {"content": "ephemeral", "category": "fact", "ttl_days": 5},
        )
    )
    assert res["ttl_days"] == 5
    doc = store.memories.find_one({"_id": res["memory_id"]})
    assert "expires_at" in doc


def test_search_returns_hits(dispatcher, store):
    store.add_memory("I prefer dark mode", category="user_pref")
    store.add_memory("I drink coffee", category="fact")
    res = json.loads(dispatcher.handle("mongo_search", {"query": "dark mode"}))
    assert res["count"] == 1
    assert res["results"][0]["content"] == "I prefer dark mode"
    assert "memory_id" in res["results"][0]


def test_search_rejects_empty_query(dispatcher):
    res = json.loads(dispatcher.handle("mongo_search", {"query": ""}))
    assert "error" in res


def test_search_filters_args(dispatcher, store):
    store.add_memory("dark mode pref", category="user_pref")
    store.add_memory("dark mode in project", category="project")
    res = json.loads(
        dispatcher.handle(
            "mongo_search", {"query": "dark mode", "category": "user_pref", "limit": 3}
        )
    )
    assert res["count"] == 1
    assert res["results"][0]["category"] == "user_pref"


def test_recall_returns_entity_neighbors(dispatcher, store):
    store.add_memory("uses dark mode", category="fact", entities=["theme"])
    store.add_memory("dark theme", category="fact", entities=["theme"])
    res = json.loads(dispatcher.handle("mongo_recall", {"entities": ["theme"]}))
    assert res["count"] >= 2


def test_recall_rejects_empty_entities(dispatcher):
    res = json.loads(dispatcher.handle("mongo_recall", {"entities": []}))
    assert "error" in res


def test_forget_deletes_immediately(dispatcher, store):
    mid = store.add_memory("disposable", category="fact")
    res = json.loads(dispatcher.handle("mongo_forget", {"memory_id": mid}))
    assert res["status"] == "removed"
    assert store.memories.find_one({"_id": mid}) is None


def test_forget_with_ttl_sets_expiry(dispatcher, store):
    mid = store.add_memory("evictable", category="fact")
    res = json.loads(
        dispatcher.handle("mongo_forget", {"memory_id": mid, "ttl_days": 1})
    )
    assert res["status"] == "ttl_set"
    assert "expires_at" in store.memories.find_one({"_id": mid})


def test_forget_handles_missing_id(dispatcher):
    res = json.loads(dispatcher.handle("mongo_forget", {"memory_id": "nope"}))
    assert res["status"] == "not_found"


def test_forget_missing_arg_returns_error(dispatcher):
    res = json.loads(dispatcher.handle("mongo_forget", {}))
    assert "error" in res


def test_profile_returns_categorized_memories(dispatcher, store):
    store.add_memory("dark mode", category="user_pref")
    store.add_memory("hermes-mongodb-memory plugin", category="project")
    store.add_memory("use MIT license", category="decision")
    res = json.loads(dispatcher.handle("mongo_profile", {}))
    assert len(res["preferences"]) == 1
    assert len(res["projects"]) == 1
    assert len(res["decisions"]) == 1


def test_reflect_groups_memories_by_entity(dispatcher, store):
    store.add_memory("a1", category="fact", entities=["e1"])
    store.add_memory("a2", category="fact", entities=["e1", "e2"])
    store.add_memory("a3", category="fact", entities=["e3"])
    res = json.loads(dispatcher.handle("mongo_reflect", {"limit": 50}))
    assert res["clusters"]
    by_entity = {c["entity"]: c["memories"] for c in res["clusters"]}
    assert len(by_entity["e1"]) == 2


def test_unknown_tool_returns_error(dispatcher):
    res = json.loads(dispatcher.handle("mongo_does_not_exist", {}))
    assert "error" in res


def test_dispatch_routes_internal_handler_naming(dispatcher, store):
    """Confirms the strip-prefix routing _handle_remember is reached."""
    res = json.loads(
        dispatcher.handle("mongo_remember", {"content": "via dispatch"})
    )
    assert res["status"] == "stored"


def test_handle_with_non_mapping_args_errors(dispatcher):
    res = json.loads(dispatcher.handle("mongo_search", "oops"))  # type: ignore[arg-type]
    assert "error" in res


def test_remember_falls_back_when_embedder_raises(store):
    """If the embedding API fails (rate limit, network), store without vector."""
    class FailingEmbedder:
        name = "fail"
        model = "fail-m"
        dim = 2

        def embed(self, text):
            raise RuntimeError("openai rate limit")

        def embed_many(self, texts):
            raise RuntimeError("openai rate limit")

    searcher = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    d = ToolDispatcher(store, searcher, embedder=FailingEmbedder())
    res = json.loads(d.handle("mongo_remember", {"content": "still works", "category": "fact"}))
    assert res["status"] == "stored"
    assert res["embedded"] is False  # graceful degradation
    doc = store.memories.find_one({"_id": res["memory_id"]})
    assert "embedding" not in doc  # no vector stored


def test_handle_with_non_dict_args_via_string(dispatcher):
    """Tool args must be a mapping; bare strings should error cleanly, not crash."""
    res = json.loads(dispatcher.handle("mongo_search", "oops"))
    assert "error" in res


def test_remember_with_unknown_tool_name(dispatcher):
    """Calling a non-existent tool returns a clean error envelope."""
    res = json.loads(dispatcher.handle("mongo_completely_made_up", {"x": 1}))
    assert "error" in res
    assert "unknown tool" in res["error"].lower()


def test_remember_with_invalid_ttl_type(dispatcher):
    """ttl_days that can't be coerced to int should error gracefully."""
    res = json.loads(
        dispatcher.handle("mongo_remember", {"content": "x", "ttl_days": "not-a-number"})
    )
    assert "error" in res


def test_search_with_invalid_min_trust_type(dispatcher, store):
    """min_trust that can't be coerced to float should error gracefully."""
    res = json.loads(
        dispatcher.handle("mongo_search", {"query": "x", "min_trust": "high"})
    )
    assert "error" in res


def test_remember_uses_embedder_when_present(store):
    class StubEmb:
        name = "stub"
        model = "stub-m"
        dim = 2

        def embed(self, text):
            return [0.5, 0.5]

        def embed_many(self, texts):
            return [[0.5, 0.5] for _ in texts]

    searcher = HybridSearcher(store, NullEmbeddingClient(), time_decay_half_life_days=0)
    d = ToolDispatcher(store, searcher, embedder=StubEmb())
    res = json.loads(d.handle("mongo_remember", {"content": "hi"}))
    assert res["embedded"] is True
    doc = store.memories.find_one({"_id": res["memory_id"]})
    assert doc["embedding"] == [0.5, 0.5]
    assert doc["embedding_model"] == "stub-m"
