"""Tests for Hermes-style plugin discovery.

We re-implement the documented discovery contract here rather than
depending on Hermes' loader directly, because this plugin must work for
any version of Hermes that follows the contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PLUGIN_ROOT / "hermes_mongodb_memory"
PACKAGE_INIT = PACKAGE_DIR / "__init__.py"
PLUGIN_YAML = PACKAGE_DIR / "plugin.yaml"


def test_init_contains_discovery_markers():
    """Hermes' loader scans the first 8KB for register_memory_provider or MemoryProvider."""
    text = PACKAGE_INIT.read_text(encoding="utf-8")[:8192]
    assert "register_memory_provider" in text or "MemoryProvider" in text


def test_register_function_signature():
    """The plugin must expose register(ctx) at top level."""
    import inspect

    from hermes_mongodb_memory import register

    sig = inspect.signature(register)
    assert "ctx" in sig.parameters


def test_register_calls_register_memory_provider():
    from hermes_mongodb_memory import register

    class FakeCtx:
        def __init__(self):
            self.providers = []

        def register_memory_provider(self, provider):
            self.providers.append(provider)

    ctx = FakeCtx()
    register(ctx)
    assert len(ctx.providers) == 1
    assert ctx.providers[0].name == "mongodb"


def test_plugin_yaml_has_required_keys():
    data = yaml.safe_load(PLUGIN_YAML.read_text(encoding="utf-8"))
    assert data["name"] == "mongodb"
    assert "version" in data
    assert "description" in data
    assert isinstance(data.get("hooks", []), list)


def test_pyproject_declares_entry_point():
    """We also expose the provider as a hermes.memory_providers entry point."""
    pyproject = PLUGIN_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "hermes.memory_providers" in text
    assert "hermes_mongodb_memory:register" in text


def test_provider_subclass_is_findable_via_module_scan():
    """Hermes' fallback path scans dir(module) for MemoryProvider subclasses."""
    import hermes_mongodb_memory as pkg

    classes = [getattr(pkg, name) for name in dir(pkg)]
    matches = [
        c for c in classes
        if isinstance(c, type) and c.__name__ == "MongoDBMemoryProvider"
    ]
    assert len(matches) == 1


@pytest.mark.parametrize(
    "marker",
    ["mongo_remember", "mongo_search", "mongo_recall", "mongo_forget", "mongo_profile", "mongo_reflect"],
)
def test_all_six_tools_discoverable(marker):
    from hermes_mongodb_memory.tools import ALL_SCHEMAS

    names = {s["function"]["name"] for s in ALL_SCHEMAS}
    assert marker in names
