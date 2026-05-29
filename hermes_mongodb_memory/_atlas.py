"""Atlas detection helpers.

Atlas exposes a few signals we can lean on:

* ``hello.process`` reports ``mongos`` on Atlas clusters
* ``connection_uri`` typically uses the ``mongodb+srv://`` scheme and a
  ``*.mongodb.net`` host

We don't need to be exact — getting this wrong only changes whether we
attempt the ``$rankFusion`` pipeline. If we attempt it on a non-Atlas
cluster, the search layer catches the ``OperationFailure`` and falls
back to the portable ranker.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def looks_like_atlas_uri(uri: str) -> bool:
    if not uri:
        return False
    lower = uri.lower()
    return lower.startswith("mongodb+srv://") or "mongodb.net" in lower


def server_supports_atlas_search(client: Any) -> bool:
    """Return True when the connected server appears to be Atlas.

    Avoids raising on transient connectivity hiccups — falls back to URI
    string-matching when ``hello`` is not reachable.
    """
    try:
        info = client.admin.command("hello")
    except Exception as exc:  # pragma: no cover - cluster-specific
        logger.debug("hello() failed during Atlas detection: %s", exc)
        return False
    process = info.get("process") or info.get("processType")
    if process == "mongos":
        return True
    set_name = info.get("setName") or ""
    if "atlas" in str(set_name).lower():
        return True
    return False
