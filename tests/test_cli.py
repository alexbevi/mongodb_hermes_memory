"""Tests for the hermes mongodb CLI subcommand."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

from hermes_mongodb_memory import cli


class FakeClient:
    def __init__(self, db):
        self._db = db

    def __getitem__(self, name):
        return self._db

    def close(self):
        pass

    @property
    def admin(self):
        from unittest.mock import MagicMock

        adm = MagicMock()
        adm.command.return_value = {"process": "mongod", "ok": 1}
        return adm


def test_register_cli_adds_three_subcommands():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd")
    cli.register_cli(sub)
    args = root.parse_args(["mongodb", "status"])
    assert args.action == "status"
    args = root.parse_args(["mongodb", "init-indexes"])
    assert args.action == "init-indexes"
    args = root.parse_args(["mongodb", "doctor"])
    assert args.action == "doctor"


def test_unknown_action_returns_nonzero(capsys):
    args = argparse.Namespace(action="bogus")
    assert cli.mongodb_command(args) == 2


def test_init_indexes_without_uri(monkeypatch, capsys):
    monkeypatch.setattr(cli.cfg, "load_config", lambda: {"connection_uri": ""})
    rc = cli.mongodb_command(argparse.Namespace(action="init-indexes"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "connection_uri not set" in err


def test_init_indexes_with_provider(mongo_db, monkeypatch, capsys):
    fake_cfg = {
        "connection_uri": "mongodb://stub",
        "database": "x",
        "embedding_provider": "none",
        "tenant_scope": "global",
    }
    monkeypatch.setattr(cli.cfg, "load_config", lambda: fake_cfg)
    with patch("pymongo.MongoClient", return_value=FakeClient(mongo_db)):
        rc = cli.mongodb_command(argparse.Namespace(action="init-indexes"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "tenant_category_recent" in out


def test_status_prints_json(mongo_db, monkeypatch, capsys):
    fake_cfg = {
        "connection_uri": "mongodb://stub",
        "database": "x",
        "embedding_provider": "none",
        "tenant_scope": "global",
    }
    monkeypatch.setattr(cli.cfg, "load_config", lambda: fake_cfg)
    with patch("pymongo.MongoClient", return_value=FakeClient(mongo_db)):
        rc = cli.mongodb_command(argparse.Namespace(action="status"))
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["database"] == "x"
    assert payload["memories"] == 0
    assert payload["atlas_detected"] is False


def test_doctor_passes_when_all_ok(mongo_db, monkeypatch, capsys):
    fake_cfg = {
        "connection_uri": "mongodb://stub",
        "database": "x",
        "embedding_provider": "none",
        "tenant_scope": "global",
    }
    monkeypatch.setattr(cli.cfg, "load_config", lambda: fake_cfg)
    with patch("pymongo.MongoClient", return_value=FakeClient(mongo_db)):
        rc = cli.mongodb_command(argparse.Namespace(action="doctor"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_doctor_flags_missing_embedding_key(mongo_db, monkeypatch, capsys):
    fake_cfg = {
        "connection_uri": "mongodb://stub",
        "database": "x",
        "embedding_provider": "openai",
        "embedding_api_key": "",
        "tenant_scope": "global",
    }
    monkeypatch.setattr(cli.cfg, "load_config", lambda: fake_cfg)
    with patch("pymongo.MongoClient", return_value=FakeClient(mongo_db)):
        rc = cli.mongodb_command(argparse.Namespace(action="doctor"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "embedding_api_key" in out
