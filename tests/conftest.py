"""Test fixtures.

Unit tests use ``mongomock`` for fast, deterministic, in-process Mongo.
Integration tests (marked with ``integration``) use a real MongoDB and
are gated by the ``MONGODB_TEST_URI`` env var.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def mongo_db():
    """Yield a clean mongomock Database for unit tests."""
    mongomock = pytest.importorskip("mongomock")
    client = mongomock.MongoClient()
    db = client["hermes_test"]
    yield db
    client.drop_database("hermes_test")


@pytest.fixture
def real_mongo_db():
    """Yield a real MongoDB Database; skips when MONGODB_TEST_URI is unset."""
    uri = os.environ.get("MONGODB_TEST_URI")
    if not uri:
        pytest.skip("MONGODB_TEST_URI not set; skipping integration test")

    from pymongo import MongoClient

    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    db_name = "hermes_test_integration"
    db = client[db_name]
    # clean before
    for coll in db.list_collection_names():
        db.drop_collection(coll)
    yield db
    # clean after
    client.drop_database(db_name)
    client.close()


@pytest.fixture
def store(mongo_db):
    """Yield a MongoStore wired to mongomock with indexes ensured."""
    from hermes_mongodb_memory.store import MongoStore

    s = MongoStore(mongo_db, tenant_id="t1")
    s.ensure_indexes()
    return s
