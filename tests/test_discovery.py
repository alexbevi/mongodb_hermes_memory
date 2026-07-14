"""Tests for Hermes-style plugin discovery.

We re-implement the documented discovery contract here rather than
depending on Hermes' loader directly, because this plugin must work for
any version of Hermes that follows the contract.
"""

from __future__ import annotations

import re
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
    assert "pymongo>=4.6,<5.0" in data["pip_dependencies"]
    assert isinstance(data.get("hooks", []), list)


def test_pyproject_declares_entry_points():
    """We expose provider entry points for future Hermes discovery paths."""
    pyproject = PLUGIN_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "hermes.memory_providers" in text
    assert "hermes_agent.plugins" in text
    assert "hermes_mongodb_memory:register" in text


def test_version_fields_match():
    import hermes_mongodb_memory

    pyproject = PLUGIN_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    pyproject_version = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert pyproject_version is not None

    data = yaml.safe_load(PLUGIN_YAML.read_text(encoding="utf-8"))
    assert pyproject_version.group(1) == hermes_mongodb_memory.__version__
    assert str(data["version"]) == hermes_mongodb_memory.__version__


def test_pyproject_declares_installer_script():
    pyproject = PLUGIN_ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    assert 'hermes-mongodb-memory = "hermes_mongodb_memory.install:main"' in text


def test_provider_class_reachable_via_attribute_access():
    """``hermes_mongodb_memory.MongoDBMemoryProvider`` resolves through __getattr__.

    Hermes' loader prefers register(ctx) (which we provide), so the fallback
    that scans dir(module) for MemoryProvider subclasses is only used when
    register() is absent. We use lazy attribute access in __init__.py to
    work around the user-plugin loader's sibling pre-import quirk.
    """
    import hermes_mongodb_memory as pkg

    cls = pkg.MongoDBMemoryProvider
    assert isinstance(cls, type)
    assert cls.__name__ == "MongoDBMemoryProvider"


@pytest.mark.parametrize(
    "marker",
    ["mongo_remember", "mongo_search", "mongo_recall", "mongo_forget", "mongo_profile", "mongo_reflect"],
)
def test_all_six_tools_discoverable(marker):
    from hermes_mongodb_memory.tools import ALL_SCHEMAS

    names = {s["function"]["name"] for s in ALL_SCHEMAS}
    assert marker in names
