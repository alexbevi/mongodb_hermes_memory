r"""End-to-end integration tests against a real MongoDB.

Gated by ``MONGODB_TEST_URI``. To run locally:

    docker compose up -d
    export MONGODB_TEST_URI="mongodb://localhost:27017/?replicaSet=rs0"
    pytest tests/test_integration.py -m integration -v

These exercise the actual Mongo aggregation pipeline (text search, TTL,
\$graphLookup) so we catch behavior that mongomock can't model.
"""

from __future__ import annotations

import json

import pytest

from hermes_mongodb_memory.embeddings import NullEmbeddingClient
from hermes_mongodb_memory.search import HybridSearcher
from hermes_mongodb_memory.store import MongoStore
from hermes_mongodb_memory.tools import ToolDispatcher

pytestmark = pytest.mark.integration


@pytest.fixture
def integration_store(real_mongo_db):
    s = MongoStore(real_mongo_db, tenant_id="integration")
    s.ensure_indexes()
    return s


def test_indexes_created_on_real_mongo(integration_store):
    names = {i["name"] for i in integration_store.memories.list_indexes()}
    assert "tenant_category_recent" in names
    assert "tenant_entities" in names
    assert "ttl_expires_at" in names
    assert "content_text" in names


def test_text_index_supports_text_search(integration_store):
    integration_store.add_memory("I prefer dark mode for night work", category="user_pref")
    integration_store.add_memory("Coffee is my fuel", category="fact")
    s = HybridSearcher(integration_store, NullEmbeddingClient(), time_decay_half_life_days=0)
    hits = s.search("dark mode")
    assert hits
    assert hits[0]["content"] == "I prefer dark mode for night work"


def test_graph_lookup_returns_neighbors(integration_store):
    integration_store.add_memory("a uses Atlas", category="fact", entities=["Atlas"])
    integration_store.add_memory("b is part of Atlas migration", category="fact", entities=["Atlas", "migration"])
    integration_store.add_memory("c uses migration tooling", category="fact", entities=["migration"])
    integration_store.add_memory("d unrelated", category="fact", entities=["other"])
    out = integration_store.related_memories(["Atlas"], max_depth=2, limit=10)
    contents = {d["content"] for d in out}
    assert {"a uses Atlas", "b is part of Atlas migration", "c uses migration tooling"} <= contents
    assert "d unrelated" not in contents


def test_full_tool_round_trip(integration_store):
    s = HybridSearcher(integration_store, NullEmbeddingClient(), time_decay_half_life_days=0)
    d = ToolDispatcher(integration_store, s)
    res = json.loads(
        d.handle("mongo_remember", {"content": "I prefer terse summaries", "category": "user_pref"})
    )
    mid = res["memory_id"]

    found = json.loads(d.handle("mongo_search", {"query": "terse summaries"}))
    assert found["count"] >= 1
    assert any(h["memory_id"] == mid for h in found["results"])

    forget = json.loads(d.handle("mongo_forget", {"memory_id": mid}))
    assert forget["status"] == "removed"
    assert integration_store.memories.find_one({"_id": mid}) is None


def test_ttl_index_eventually_evicts(integration_store):
    """The TTL monitor runs every 60s by default — we don't actually wait
    for eviction. We assert the TTL field is set so MongoDB will evict it,
    which is the contract we ship to users.
    """
    mid = integration_store.add_memory("ephemeral", category="fact", ttl_days=1)
    doc = integration_store.memories.find_one({"_id": mid})
    assert "expires_at" in doc
    # confirm the TTL index actually mentions expireAfterSeconds=0
    ttl_index = next(
        i for i in integration_store.memories.list_indexes() if i["name"] == "ttl_expires_at"
    )
    assert ttl_index["expireAfterSeconds"] == 0
