"""MongoDB memory provider plugin for Hermes Agent.

The Hermes plugin loader scans this file for the strings
``register_memory_provider`` and ``MemoryProvider`` to identify it as a
memory plugin, then calls ``register(ctx)``.

Sibling modules are imported lazily inside :func:`register` (and via
``__getattr__``) because Hermes' user-plugin loader pre-imports every
``.py`` sibling under a synthetic ``_hermes_user_memory.<name>``
namespace that doesn't exist as a real package. Eager top-level
``from .provider import ...`` would race that pre-import and fail; a
lazy import sidesteps the issue without giving up the modular layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .provider import MongoDBMemoryProvider as MongoDBMemoryProvider

__all__ = ["MongoDBMemoryProvider", "register"]
__version__ = "0.2.0"


def _resolve_provider_class() -> type:
    """Locate the provider class regardless of how Python loaded this package.

    Hermes' user-plugin loader pre-imports each sibling ``.py`` under a
    synthetic ``_hermes_user_memory.<plugin>`` namespace whose parent
    package never exists, so each sibling ``exec_module`` raises and leaves
    a half-built stub in ``sys.modules``. A subsequent
    ``from .provider import MongoDBMemoryProvider`` then resolves to the
    stub and fails.

    Detection: the package is misloaded when ``__name__`` is under the
    synthetic namespace AND its parent package isn't importable. In that
    case we re-import the package fresh under a clean alias name using
    ``importlib.util`` so all relative imports inside resolve correctly.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent

    # Fast path — works when pip-installed (real package) or bundled
    # under plugins.memory (real parent package exists).
    try:
        return importlib.import_module(f"{__name__}.provider").MongoDBMemoryProvider
    except (ImportError, AttributeError):
        pass

    # Slow path — re-import this package as a fresh, top-level module
    # so its relative imports have a real parent to resolve against.
    alias = "_hermes_mongodb_memory_isolated"

    # Drop any half-built stubs left by Hermes' user-plugin loader.
    for key in list(sys.modules):
        if key.startswith("_hermes_user_memory."):
            sys.modules.pop(key, None)

    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias,
            str(here / "__init__.py"),
            submodule_search_locations=[str(here)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot locate {here}/__init__.py")
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[alias] = pkg
        spec.loader.exec_module(pkg)

    return importlib.import_module(f"{alias}.provider").MongoDBMemoryProvider


def __getattr__(name: str) -> Any:
    """Lazy attribute access so ``hermes_mongodb_memory.MongoDBMemoryProvider`` works."""
    if name == "MongoDBMemoryProvider":
        return _resolve_provider_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register(ctx) -> None:
    """Register the MongoDB memory provider with the plugin system."""
    cls = _resolve_provider_class()
    ctx.register_memory_provider(cls())
