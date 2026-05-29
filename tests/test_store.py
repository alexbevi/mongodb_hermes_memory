"""Tests for MongoStore — collections, indexes, CRUD, TTL handling."""

from __future__ import annotations

import datetime as dt
import time

import pytest

from hermes_mongodb_memory.store import (
    MEMORIES,
    TURNS,
    MongoStore,
    normalize_entity,
    utcnow,
)


def _index_names(coll):
    return {info["name"] for info in coll.list_indexes()}


def test_normalize_entity_collapses_punctuation_and_case():
    assert normalize_entity("Dark-Mode") == "dark mode"
    assert normalize_entity("  Dark   Mode  ") == "dark   mode" or normalize_entity("  Dark   Mode  ") == "dark mode"


def test_ensure_indexes_creates_expected_indexes(store):
    names = _index_names(store.memories)
    assert "tenant_category_recent" in names
    assert "tenant_entities" in names
    assert "tenant_tags" in names
    assert "ttl_expires_at" in names

    turn_names = _index_names(store.turns)
    assert "tenant_session_turn" in turn_names
    assert "turns_ttl" in turn_names

    entity_names = _index_names(store.entities)
    assert "tenant_entity_unique" in entity_names


def test_ensure_indexes_is_idempotent(store):
    store.ensure_indexes()
    store.ensure_indexes()
    assert "tenant_category_recent" in _index_names(store.memories)


def test_add_memory_inserts_and_returns_id(store):
    mid = store.add_memory(
        "I prefer dark mode",
        category="user_pref",
        entities=["Dark-Mode"],
        tags=["ui"],
    )
    assert isinstance(mid, str) and len(mid) > 0
    doc = store.memories.find_one({"_id": mid})
    assert doc["content"] == "I prefer dark mode"
    assert doc["category"] == "user_pref"
    assert doc["entities"] == ["dark mode"]
    assert doc["tags"] == ["ui"]
    assert doc["tenant_id"] == "t1"
    assert doc["trust"] == pytest.approx(0.5)


def test_add_memory_rejects_empty_content(store):
    with pytest.raises(ValueError):
        store.add_memory("   ", category="fact")


def test_add_memory_rejects_unknown_category(store):
    with pytest.raises(ValueError):
        store.add_memory("hello", category="invalid")


def test_add_memory_with_ttl_sets_expires_at(store):
    mid = store.add_memory("ephemeral", category="fact", ttl_days=7)
    doc = store.memories.find_one({"_id": mid})
    assert "expires_at" in doc
    # mongomock strips tz on round-trip; compare against the stored created_at.
    delta = doc["expires_at"] - doc["created_at"]
    assert delta == dt.timedelta(days=7)


def test_add_memory_with_embedding_stores_vector(store):
    mid = store.add_memory(
        "vector ready",
        category="fact",
        embedding=[0.1, 0.2, 0.3],
        embedding_model="test-emb",
    )
    doc = store.memories.find_one({"_id": mid})
    assert doc["embedding"] == [0.1, 0.2, 0.3]
    assert doc["embedding_model"] == "test-emb"


def test_remove_memory_returns_true_when_present(store):
    mid = store.add_memory("disposable", category="general")
    assert store.remove_memory(mid) is True
    assert store.memories.find_one({"_id": mid}) is None


def test_remove_memory_respects_tenant(store, mongo_db):
    mid = store.add_memory("mine", category="fact")
    other = MongoStore(mongo_db, tenant_id="other")
    assert other.remove_memory(mid) is False
    assert store.memories.find_one({"_id": mid}) is not None


def test_expire_memory_sets_short_ttl(store):
    mid = store.add_memory("evictable", category="fact")
    assert store.expire_memory(mid, ttl_days=1) is True
    doc = store.memories.find_one({"_id": mid})
    assert "expires_at" in doc


def test_expire_memory_with_zero_deletes(store):
    mid = store.add_memory("evictable", category="fact")
    assert store.expire_memory(mid, ttl_days=0) is True
    assert store.memories.find_one({"_id": mid}) is None


def test_list_memories_filters_by_category(store):
    store.add_memory("pref1", category="user_pref")
    store.add_memory("project1", category="project")
    prefs = store.list_memories(category="user_pref")
    assert len(prefs) == 1
    assert prefs[0]["content"] == "pref1"


def test_list_memories_filters_by_entity_and_min_trust(store):
    store.add_memory("about coffee", category="fact", entities=["coffee"])
    store.add_memory("about tea", category="fact", entities=["tea"])
    coffee = store.list_memories(entities=["Coffee"])
    assert len(coffee) == 1
    assert coffee[0]["content"] == "about coffee"

    none_high_trust = store.list_memories(min_trust=0.9)
    assert none_high_trust == []


def test_count_memories_scopes_by_tenant(store, mongo_db):
    store.add_memory("a", category="fact")
    store.add_memory("b", category="fact")
    other = MongoStore(mongo_db, tenant_id="other")
    other.ensure_indexes()
    other.add_memory("c", category="fact")
    assert store.count_memories() == 2
    assert other.count_memories() == 1


def test_record_feedback_adjusts_trust_and_counters(store):
    mid = store.add_memory("rate-me", category="fact")
    res = store.record_feedback(mid, helpful=True)
    assert res["trust"] > 0.5
    res2 = store.record_feedback(mid, helpful=False)
    assert res2["trust"] < res["trust"]
    doc = store.memories.find_one({"_id": mid})
    assert doc["helpful_count"] == 1
    assert doc["unhelpful_count"] == 1


def test_record_feedback_clamps_to_unit_interval(store):
    mid = store.add_memory("clamp", category="fact")
    for _ in range(50):
        store.record_feedback(mid, helpful=True)
    assert store.memories.find_one({"_id": mid})["trust"] == pytest.approx(1.0)


def test_record_feedback_returns_none_for_missing(store):
    assert store.record_feedback("missing", helpful=True) is None


def test_append_turn_writes_with_ttl(store):
    store.append_turn(session_id="s1", turn_idx=1, role="user", content="hi", ttl_days=2)
    doc = store.turns.find_one({"session_id": "s1", "turn_idx": 1})
    assert doc["role"] == "user"
    assert doc["content"] == "hi"
    assert "expires_at" in doc


def test_append_turn_skips_empty_content(store):
    store.append_turn(session_id="s1", turn_idx=1, role="user", content="")
    assert store.turns.count_documents({}) == 0


def test_next_turn_idx(store):
    assert store.next_turn_idx("s1") == 1
    store.append_turn(session_id="s1", turn_idx=1, role="user", content="hi")
    store.append_turn(session_id="s1", turn_idx=2, role="assistant", content="hey")
    assert store.next_turn_idx("s1") == 3


def test_related_memories_returns_entity_neighbors(store):
    store.add_memory("uses dark mode", category="fact", entities=["theme"])
    store.add_memory("dark theme everywhere", category="fact", entities=["theme"])
    store.add_memory("unrelated", category="fact", entities=["fish"])
    out = store.related_memories(["theme"])
    assert len(out) >= 2
    contents = {d["content"] for d in out}
    assert "uses dark mode" in contents
    assert "unrelated" not in contents


def test_upsert_profile_round_trip(store):
    store.upsert_profile("a person", ["dark mode"])
    p = store.get_profile()
    assert p["summary"] == "a person"
    assert p["preferences"] == ["dark mode"]


def test_entities_collection_tracks_mentions(store):
    store.add_memory("a", category="fact", entities=["coffee"])
    store.add_memory("b", category="fact", entities=["coffee"])
    doc = store.entities.find_one({"normalized": "coffee"})
    assert doc["mention_count"] == 2
    assert doc["tenant_id"] == "t1"


def test_time_decay_weight_zero_half_life_returns_one(store):
    now = utcnow()
    assert store.time_decay_weight(now, 0) == 1.0


def test_time_decay_weight_decays_over_time(store):
    past = utcnow() - dt.timedelta(days=30)
    weight = store.time_decay_weight(past, half_life_days=30)
    assert weight == pytest.approx(0.5, rel=0.01)
