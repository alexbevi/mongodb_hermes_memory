"""Tests for the pluggable embedding clients."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_mongodb_memory import embeddings


def test_null_client_returns_empty_vectors():
    c = embeddings.NullEmbeddingClient()
    assert c.embed("hi") == []
    assert c.embed_many(["a", "b", "c"]) == [[], [], []]
    assert c.dim == 0
    assert c.name == "none"


def test_make_embedding_client_default_is_null():
    c = embeddings.make_embedding_client({})
    assert isinstance(c, embeddings.NullEmbeddingClient)


def test_make_embedding_client_explicit_none_is_null():
    c = embeddings.make_embedding_client({"embedding_provider": "none"})
    assert isinstance(c, embeddings.NullEmbeddingClient)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        embeddings.make_embedding_client({"embedding_provider": "weirdvec"})


def test_openai_client_requires_api_key(monkeypatch):
    # Stub the openai module so the import succeeds even if not installed.
    monkeypatch.setitem(__import__("sys").modules, "openai", MagicMock())
    with pytest.raises(ValueError):
        embeddings.OpenAIEmbeddingClient(api_key="", model="m", dim=128)


def test_openai_client_embed_calls_sdk(monkeypatch):
    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_openai.OpenAI.return_value = fake_client

    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    fake_client.embeddings.create.return_value = response

    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    c = embeddings.OpenAIEmbeddingClient(api_key="k", model="text-embedding-3-small", dim=2)
    out = c.embed_many(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]

    call_kwargs = fake_client.embeddings.create.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-3-small"
    assert call_kwargs["input"] == ["a", "b"]
    assert call_kwargs["dimensions"] == 2


def test_openai_client_skips_dimensions_for_legacy_models(monkeypatch):
    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.0])]
    fake_client.embeddings.create.return_value = response
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    c = embeddings.OpenAIEmbeddingClient(api_key="k", model="text-embedding-ada-002", dim=1536)
    c.embed("x")

    kwargs = fake_client.embeddings.create.call_args.kwargs
    assert "dimensions" not in kwargs


def test_voyage_client_requires_api_key(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "voyageai", MagicMock())
    with pytest.raises(ValueError):
        embeddings.VoyageEmbeddingClient(api_key="", model="voyage-3", dim=1024)


def test_voyage_client_embed_calls_sdk(monkeypatch):
    fake_voyage_module = MagicMock()
    fake_client = MagicMock()
    fake_voyage_module.Client.return_value = fake_client

    fake_resp = MagicMock()
    fake_resp.embeddings = [[0.1, 0.2], [0.3, 0.4]]
    fake_client.embed.return_value = fake_resp

    monkeypatch.setitem(__import__("sys").modules, "voyageai", fake_voyage_module)

    c = embeddings.VoyageEmbeddingClient(api_key="k", model="voyage-3", dim=2)
    out = c.embed_many(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.embed.assert_called_once_with(["a", "b"], model="voyage-3", input_type="document")


def test_make_embedding_client_routes_to_openai(monkeypatch):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

    c = embeddings.make_embedding_client(
        {
            "embedding_provider": "openai",
            "embedding_api_key": "k",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        }
    )
    assert isinstance(c, embeddings.OpenAIEmbeddingClient)
    assert c.model == "text-embedding-3-small"
    assert c.dim == 1536


def test_make_embedding_client_routes_to_voyage(monkeypatch):
    fake_voyage = MagicMock()
    fake_voyage.Client.return_value = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "voyageai", fake_voyage)

    c = embeddings.make_embedding_client(
        {
            "embedding_provider": "voyage",
            "embedding_api_key": "k",
            "embedding_model": "voyage-3",
            "embedding_dim": 1024,
        }
    )
    assert isinstance(c, embeddings.VoyageEmbeddingClient)


def test_embed_many_with_empty_input_returns_empty_list(monkeypatch):
    fake_openai = MagicMock()
    fake_client = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    c = embeddings.OpenAIEmbeddingClient(api_key="k", model="text-embedding-3-small", dim=2)
    assert c.embed_many([]) == []
    fake_client.embeddings.create.assert_not_called()
