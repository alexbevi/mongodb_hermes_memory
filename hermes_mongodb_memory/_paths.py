"""Path helpers shared by the MongoDB provider installer."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path

PLUGIN_NAME = "mongodb"


def _hermes_home_from_hermes() -> Path | None:
    """Return Hermes' configured home path when Hermes is importable."""
    try:
        hermes_constants = import_module("hermes_constants")
    except Exception:
        return None

    get_hermes_home = getattr(hermes_constants, "get_hermes_home", None)
    if not callable(get_hermes_home):
        return None
    return Path(get_hermes_home()).expanduser()


def _platform_default_hermes_home() -> Path:
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        if local_appdata:
            return Path(local_appdata).expanduser() / "hermes"
        return Path.home() / "AppData" / "Local" / "hermes"
    return Path("~/.hermes").expanduser()


def resolve_hermes_home(hermes_home_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the Hermes home directory using Hermes-compatible precedence."""
    if hermes_home_path:
        return Path(hermes_home_path).expanduser()

    hermes_home = _hermes_home_from_hermes()
    if hermes_home is not None:
        return hermes_home

    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser()

    return _platform_default_hermes_home()


def plugin_target_dir(hermes_home_path: str | os.PathLike[str] | None = None) -> Path:
    """Return the user-installed Hermes memory plugin destination."""
    return resolve_hermes_home(hermes_home_path) / "plugins" / PLUGIN_NAME
