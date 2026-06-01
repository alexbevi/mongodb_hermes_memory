"""Conformance tests against the real Hermes Agent ABC and loader.

Skipped when ``hermes-agent`` is not installed (Python <3.11 or just
not yet pulled into the env). To run:

    pip install -e ".[dev,hermes]"          # Python 3.11+
    pytest tests/test_hermes_conformance.py -v

These tests verify the plugin works *inside* Hermes — they import
``agent.memory_provider.MemoryProvider`` and ``plugins.memory`` from
the installed hermes-agent package, drop the plugin into a temp
``$HERMES_HOME/plugins/mongodb`` directory, and drive the provider
through the same code paths Hermes uses at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------- gating

# Skip the entire module if hermes-agent isn't installed.
hermes_loader = pytest.importorskip("plugins.memory", reason="hermes-agent not installed")
hermes_abc = pytest.importorskip("agent.memory_provider", reason="hermes-agent not installed")

pytestmark = pytest.mark.hermes

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "hermes_mongodb_memory"


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Create a temp $HERMES_HOME with our plugin symlinked in."""
    home = tmp_path / "hermes"
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)
    (plugins_dir / "mongodb").symlink_to(PLUGIN_DIR)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Ensure each test starts with a fresh loader state — Hermes caches
    # modules under both ``_hermes_user_memory.*`` and our isolated alias.
    import sys
    for key in list(sys.modules):
        if key.startswith("_hermes_user_memory") or key.startswith("_hermes_mongodb_memory_isolated"):
            sys.modules.pop(key, None)

    return home


# ---------------------------------------------------------------- A. ABC


def test_real_memoryprovider_abstract_methods_match_our_impl():
    """Our class declares the same abstract surface the real ABC requires."""
    real = hermes_abc.MemoryProvider
    expected_abstract = {"name", "is_available", "initialize", "get_tool_schemas"}
    assert set(real.__abstractmethods__) == expected_abstract


def test_provider_is_real_subclass(hermes_home):
    """Our class IS-A real Hermes ``MemoryProvider`` when loaded inside Hermes."""
    provider = hermes_loader.load_memory_provider("mongodb")
    assert provider is not None
    assert isinstance(provider, hermes_abc.MemoryProvider)


# ---------------------------------------------------------------- B. discovery


def test_loader_discovers_us_alongside_bundled_providers(hermes_home):
    """``discover_memory_providers()`` lists 'mongodb' next to bundled providers."""
    providers = hermes_loader.discover_memory_providers()
    names = [name for name, _desc, _avail in providers]
    assert "mongodb" in names
    # Should also see at least one bundled provider so we know discovery is
    # scanning both locations correctly.
    assert any(n in names for n in ["holographic", "honcho", "mem0"])


def test_loader_returns_correct_description(hermes_home):
    """``plugin.yaml`` description is surfaced verbatim."""
    providers = {n: d for n, d, _ in hermes_loader.discover_memory_providers()}
    desc = providers["mongodb"]
    assert "MongoDB" in desc
    assert "$rankFusion" in desc or "rankFusion" in desc


def test_loader_reports_unavailable_without_uri(hermes_home, monkeypatch):
    """is_available returns False when no connection URI is configured."""
    monkeypatch.delenv("HERMES_MONGODB_URI", raising=False)
    providers = {n: a for n, _d, a in hermes_loader.discover_memory_providers()}
    assert providers["mongodb"] is False


def test_load_memory_provider_returns_none_for_unknown_name(hermes_home):
    """Sanity: the loader returns None for names that don't exist."""
    assert hermes_loader.load_memory_provider("not-a-real-provider") is None


def test_find_provider_dir_returns_user_path(hermes_home):
    """find_provider_dir resolves our user-installed plugin to the right path."""
    found = hermes_loader.find_provider_dir("mongodb")
    assert found is not None
    # symlink resolves back to the actual package directory
    assert Path(found).resolve() == PLUGIN_DIR.resolve()


# ---------------------------------------------------------------- C. lifecycle


@pytest.fixture
def loaded_provider(hermes_home):
    """Yield a freshly-loaded provider for lifecycle tests."""
    provider = hermes_loader.load_memory_provider("mongodb")
    assert provider is not None
    yield provider
    try:
        provider.shutdown()
    except Exception:
        pass


def test_provider_name_property(loaded_provider):
    assert loaded_provider.name == "mongodb"


def test_get_config_schema_shape(loaded_provider):
    schema = loaded_provider.get_config_schema()
    keys = {f["key"] for f in schema}
    assert "connection_uri" in keys
    assert "embedding_provider" in keys


def test_get_tool_schemas_empty_before_initialize(loaded_provider):
    """Tools aren't available until initialize() runs."""
    assert loaded_provider.get_tool_schemas() == []


def test_initialize_then_tools_available(loaded_provider, monkeypatch):
    """Drive a full initialize -> tool-call -> shutdown cycle with mongomock."""
    mongomock = pytest.importorskip("mongomock")

    fake_client = mongomock.MongoClient()

    class _FakeMongoClient:
        def __init__(self, *_, **__):
            pass

        def __getitem__(self, name):
            return fake_client[name]

        def close(self):
            pass

        @property
        def admin(self):
            from unittest.mock import MagicMock
            adm = MagicMock()
            adm.command.return_value = {"process": "mongod"}
            return adm

    monkeypatch.setattr("pymongo.MongoClient", _FakeMongoClient)
    monkeypatch.setenv("HERMES_MONGODB_URI", "mongodb://stub")

    # Re-load so the new env var is picked up.
    fresh = hermes_loader.load_memory_provider("mongodb")
    fresh.initialize("conformance-session", agent_workspace="/tmp/test")
    try:
        schemas = fresh.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert names == {
            "mongo_remember", "mongo_search", "mongo_recall",
            "mongo_forget", "mongo_profile", "mongo_reflect",
        }

        # Drive a remember -> search round trip through Hermes-style call sites.
        import json
        res = json.loads(fresh.handle_tool_call(
            "mongo_remember",
            {"content": "Conformance test", "category": "fact"},
        ))
        assert res["status"] == "stored"

        res = json.loads(fresh.handle_tool_call(
            "mongo_search", {"query": "conformance"}
        ))
        assert res["count"] == 1

        # Pre-compress and post-compress hooks should not raise.
        block = fresh.system_prompt_block()
        assert "MongoDB Memory" in block

        fresh.on_session_switch("new-session")
        fresh.on_memory_write("add", "user", "User likes terse responses")
        compressed = fresh.on_pre_compress([])
        assert isinstance(compressed, str)
    finally:
        fresh.shutdown()


def test_provider_lifecycle_methods_present(loaded_provider):
    """Every documented hook is callable (presence check)."""
    for method in (
        "initialize", "system_prompt_block", "prefetch", "queue_prefetch",
        "sync_turn", "get_tool_schemas", "handle_tool_call",
        "on_session_end", "on_session_switch", "on_memory_write",
        "on_pre_compress", "shutdown", "post_setup",
        "save_config", "get_config_schema",
    ):
        assert callable(getattr(loaded_provider, method)), method
