from __future__ import annotations

import sys
import types
from pathlib import Path

from hermes_mongodb_memory import _paths


def test_resolve_hermes_home_prefers_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env-home"))
    assert _paths.resolve_hermes_home(tmp_path / "explicit") == tmp_path / "explicit"


def test_resolve_hermes_home_prefers_hermes_constant(tmp_path, monkeypatch):
    hermes_home = tmp_path / "from-hermes"
    module = types.SimpleNamespace(get_hermes_home=lambda: hermes_home)
    monkeypatch.setitem(sys.modules, "hermes_constants", module)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env-home"))

    assert _paths.resolve_hermes_home() == hermes_home


def test_resolve_hermes_home_uses_env_when_hermes_absent(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "hermes_constants", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env-home"))

    assert _paths.resolve_hermes_home() == tmp_path / "env-home"


def test_plugin_target_dir_points_to_mongodb_plugin(tmp_path):
    assert _paths.plugin_target_dir(tmp_path) == Path(tmp_path) / "plugins" / "mongodb"
