"""Tests for MongoDBMemoryProvider lifecycle (mongomock-backed)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_mongodb_memory.provider import MongoDBMemoryProvider


class FakeClient:
    def __init__(self, db):
        self._db = db
        self.closed = False

    def __getitem__(self, name):
        return self._db

    def close(self):
        self.closed = True

    @property
    def admin(self):
        adm = MagicMock()
        adm.command.return_value = {"process": "mongod"}
        return adm


@pytest.fixture
def provider(mongo_db):
    fake = FakeClient(mongo_db)
    base_cfg = {
        "connection_uri": "mongodb://stub",
        "database": "hermes_memory",
        "embedding_provider": "none",
        "tenant_scope": "global",
        "time_decay_half_life_days": 0,
        "default_ttl_days": 0,
        "prefetch_limit": 5,
        "turn_ttl_days": 0,
    }
    with patch("pymongo.MongoClient", return_value=fake):
        p = MongoDBMemoryProvider(config=base_cfg)
        p.initialize("session-1")
        yield p
        p.shutdown()


def test_name_is_mongodb():
    assert MongoDBMemoryProvider({}).name == "mongodb"


def test_is_available_requires_uri():
    p = MongoDBMemoryProvider({"connection_uri": ""})
    assert p.is_available() is False
    p2 = MongoDBMemoryProvider({"connection_uri": "mongodb://localhost"})
    assert p2.is_available() is True


def test_get_tool_schemas_returns_six(provider):
    schemas = provider.get_tool_schemas()
    assert len(schemas) == 6


def test_initialize_then_handle_tool_call(provider):
    import json

    res = json.loads(
        provider.handle_tool_call(
            "mongo_remember", {"content": "I prefer dark mode", "category": "user_pref"}
        )
    )
    assert res["status"] == "stored"
    res2 = json.loads(provider.handle_tool_call("mongo_search", {"query": "dark mode"}))
    assert res2["count"] == 1


def test_handle_tool_call_before_initialize_returns_error():
    p = MongoDBMemoryProvider({"connection_uri": "mongodb://stub"})
    out = p.handle_tool_call("mongo_search", {"query": "x"})
    assert "error" in out.lower() or "not initialized" in out.lower()


def test_system_prompt_block_changes_with_count(provider):
    assert "No memories yet" in provider.system_prompt_block()
    provider.handle_tool_call("mongo_remember", {"content": "fact one", "category": "fact"})
    assert "1 memories" in provider.system_prompt_block()


def test_prefetch_returns_formatted_results(provider):
    provider.handle_tool_call(
        "mongo_remember", {"content": "I prefer dark mode", "category": "user_pref"}
    )
    out = provider.prefetch("dark mode")
    assert "## MongoDB Memory" in out
    assert "dark mode" in out


def test_prefetch_empty_when_no_match(provider):
    assert provider.prefetch("nothing") == ""


def test_queue_prefetch_runs_in_background(provider):
    provider.handle_tool_call(
        "mongo_remember", {"content": "I prefer dark mode", "category": "user_pref"}
    )
    provider.queue_prefetch("dark mode")
    # join via the prefetch() public path
    out = provider.prefetch("dark mode")
    assert "dark mode" in out


def test_sync_turn_writes_turns_async(provider):
    provider.sync_turn("hello", "world", session_id="s")
    # wait for daemon to finish
    if provider._sync_thread:
        provider._sync_thread.join(timeout=2.0)
    docs = list(provider._store.turns.find({"session_id": "s"}))
    roles = sorted(d["role"] for d in docs)
    assert roles == ["assistant", "user"]


def test_sync_turn_skips_when_breaker_open(provider):
    provider._failures = 10
    provider._breaker_opened_at = time.time()
    provider.sync_turn("a", "b", session_id="s")
    if provider._sync_thread:
        provider._sync_thread.join(timeout=1.0)
    assert provider._store.turns.count_documents({}) == 0


def test_record_failure_opens_breaker(provider):
    for _ in range(5):
        provider._record_failure()
    assert provider._is_breaker_open() is True
    provider._record_success()
    assert provider._is_breaker_open() is False


def test_on_session_end_with_auto_extract_inserts_memories(mongo_db):
    fake = FakeClient(mongo_db)
    cfg = {
        "connection_uri": "mongodb://stub",
        "database": "hermes_memory",
        "embedding_provider": "none",
        "tenant_scope": "global",
        "auto_extract": True,
        "time_decay_half_life_days": 0,
    }
    with patch("pymongo.MongoClient", return_value=fake):
        p = MongoDBMemoryProvider(config=cfg)
        p.initialize("s1")
        p.on_session_end(
            [{"role": "user", "content": "I prefer dark mode for all the things."}]
        )
        prefs = p._store.list_memories(category="user_pref")
        assert len(prefs) == 1


def test_on_session_end_skips_when_auto_extract_off(provider):
    provider.on_session_end([{"role": "user", "content": "I prefer dark mode for all"}])
    assert provider._store.count_memories() == 0


def test_on_memory_write_mirrors_user_pref(provider):
    provider.on_memory_write("add", "user", "User likes terse responses", metadata={"src": "x"})
    docs = list(provider._store.list_memories(category="user_pref"))
    assert len(docs) == 1
    assert docs[0]["source"] == "mirror:user"


def test_on_memory_write_ignores_remove(provider):
    provider.on_memory_write("remove", "memory", "x", metadata=None)
    assert provider._store.count_memories() == 0


def test_on_session_switch_clears_prefetch(provider):
    with provider._prefetch_lock:
        provider._prefetch_result = "stale"
    provider.on_session_switch("new-id")
    with provider._prefetch_lock:
        assert provider._prefetch_result == ""
    assert provider._session_id == "new-id"


def test_on_pre_compress_returns_high_trust_memories(provider):
    provider.handle_tool_call(
        "mongo_remember", {"content": "I prefer dark mode", "category": "user_pref"}
    )
    out = provider.on_pre_compress([])
    assert "I prefer dark mode" in out


def test_shutdown_closes_client(provider):
    provider.shutdown()
    # second shutdown should not raise
    provider.shutdown()
    assert provider._store is None


def test_save_config_redacts_secrets(provider, tmp_path):
    provider.save_config({"database": "x", "connection_uri": "leaked"}, str(tmp_path))
    import json

    on_disk = json.loads((tmp_path / "mongodb.json").read_text())
    assert "connection_uri" not in on_disk
    assert on_disk["database"] == "x"


def test_post_setup_returns_ok_on_success(mongo_db, tmp_path):
    fake = FakeClient(mongo_db)
    cfg = {"connection_uri": "mongodb://stub", "database": "x", "embedding_provider": "none"}
    with patch("pymongo.MongoClient", return_value=fake):
        p = MongoDBMemoryProvider(config=cfg)
        out = p.post_setup(str(tmp_path), config=cfg)
        assert out["ok"] is True


def test_post_setup_returns_error_when_no_uri(tmp_path):
    p = MongoDBMemoryProvider(config={"connection_uri": ""})
    out = p.post_setup(str(tmp_path), config={})
    assert out["ok"] is False


def test_tenant_derivation_per_workspace(mongo_db):
    fake = FakeClient(mongo_db)
    cfg = {
        "connection_uri": "mongodb://stub",
        "database": "x",
        "embedding_provider": "none",
        "tenant_scope": "per-workspace",
    }
    with patch("pymongo.MongoClient", return_value=fake):
        p = MongoDBMemoryProvider(config=cfg)
        p.initialize("s1", agent_workspace="/Users/alex/Workspace/foo")
        assert p._tenant_id.startswith("ws:")
        p.shutdown()


def test_tenant_derivation_per_profile(mongo_db):
    fake = FakeClient(mongo_db)
    cfg = {
        "connection_uri": "mongodb://stub",
        "database": "x",
        "embedding_provider": "none",
        "tenant_scope": "per-profile",
    }
    with patch("pymongo.MongoClient", return_value=fake):
        p = MongoDBMemoryProvider(config=cfg)
        p.initialize("s1", agent_identity="alex")
        assert p._tenant_id == "profile:alex"
        p.shutdown()
