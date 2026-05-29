"""Pluggable embedding clients.

Each backend is a thin wrapper that lazily imports its SDK so the plugin
runs in BM25-only mode without any extra dependency installed.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingClient(Protocol):
    """Minimal interface every embedding backend must implement."""

    name: str
    model: str
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class NullEmbeddingClient:
    """Used when ``embedding_provider == 'none'``; produces no embeddings."""

    name = "none"
    model = ""
    dim = 0

    def embed(self, text: str) -> list[float]:
        return []

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class OpenAIEmbeddingClient:
    """OpenAI text-embedding-3-* client; SDK imported lazily."""

    name = "openai"

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        if not api_key:
            raise ValueError("OpenAI embedding client requires api_key")
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Install hermes-mongodb-memory[openai]"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = int(dim)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, object] = {"model": self.model, "input": texts}
        # The 3-series models accept `dimensions`; older models don't.
        if self.dim and self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dim
        resp = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in resp.data]


class VoyageEmbeddingClient:
    """Voyage AI client; SDK imported lazily."""

    name = "voyage"

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        if not api_key:
            raise ValueError("Voyage embedding client requires api_key")
        try:
            import voyageai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "voyageai package not installed. Install hermes-mongodb-memory[voyage]"
            ) from exc
        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dim = int(dim)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embed(texts, model=self.model, input_type="document")
        return list(resp.embeddings)


def make_embedding_client(config: dict) -> EmbeddingClient:
    """Construct the embedding client described by ``config``.

    Returns :class:`NullEmbeddingClient` when no provider is configured.
    Falls back to the null client (with a warning) on misconfiguration so
    a missing API key disables vector search rather than crashing the
    plugin — the doctor subcommand surfaces this condition explicitly.
    """
    provider = (config.get("embedding_provider") or "none").strip().lower()
    if provider == "none":
        return NullEmbeddingClient()

    api_key = config.get("embedding_api_key", "") or ""
    model = config.get("embedding_model", "") or ""
    dim = config.get("embedding_dim", 0) or 0

    try:
        if provider == "openai":
            return OpenAIEmbeddingClient(api_key=api_key, model=model or "text-embedding-3-small", dim=dim or 1536)
        if provider == "voyage":
            return VoyageEmbeddingClient(api_key=api_key, model=model or "voyage-3", dim=dim or 1024)
    except (ValueError, RuntimeError) as exc:
        logger.warning("embedding provider %r unusable, falling back to BM25-only: %s", provider, exc)
        return NullEmbeddingClient()

    logger.warning("unknown embedding_provider: %r — falling back to BM25-only", provider)
    return NullEmbeddingClient()
