from __future__ import annotations

from pathlib import Path

import pytest

from hermes_mongodb_memory import install


def test_install_plugin_copies_provider_to_hermes_plugins(tmp_path):
    target = install.install_plugin(hermes_home_path=tmp_path)

    assert target == tmp_path / "plugins" / "mongodb"
    assert (target / "__init__.py").is_file()
    assert (target / "plugin.yaml").is_file()
    assert not (target / "__pycache__").exists()
    assert install.is_installed(hermes_home_path=tmp_path)


def test_install_plugin_refuses_existing_target_without_force(tmp_path):
    install.install_plugin(hermes_home_path=tmp_path)

    with pytest.raises(FileExistsError):
        install.install_plugin(hermes_home_path=tmp_path)


def test_install_plugin_force_replaces_existing_target(tmp_path):
    target = install.install_plugin(hermes_home_path=tmp_path)
    stale = target / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    install.install_plugin(hermes_home_path=tmp_path, force=True)

    assert not stale.exists()
    assert (target / "__init__.py").is_file()


def test_uninstall_plugin_removes_target(tmp_path):
    target = install.install_plugin(hermes_home_path=tmp_path)

    removed = install.uninstall_plugin(hermes_home_path=tmp_path)

    assert removed == target
    assert not target.exists()
    assert not install.is_installed(hermes_home_path=tmp_path)


def test_status_cli_reports_missing_install(tmp_path, capsys):
    rc = install.main(["--hermes-home", str(tmp_path), "status"])

    assert rc == 1
    assert "not installed" in capsys.readouterr().out


def test_install_cli_defaults_to_install(tmp_path, capsys):
    rc = install.main(["--hermes-home", str(tmp_path)])

    assert rc == 0
    assert install.is_installed(hermes_home_path=tmp_path)
    assert "Next steps" in capsys.readouterr().out


def test_uninstall_cli_removes_install(tmp_path, capsys):
    install.install_plugin(hermes_home_path=tmp_path)

    rc = install.main(["--hermes-home", str(tmp_path), "uninstall"])

    assert rc == 0
    assert not install.is_installed(hermes_home_path=tmp_path)
    assert "Removed" in capsys.readouterr().out


def test_ignore_copy_names_filters_caches_and_bytecode():
    ignored = install._ignore_copy_names(
        str(Path.cwd()),
        ["__pycache__", ".pytest_cache", "module.pyc", "module.py", "plugin.yaml"],
    )

    assert ignored == {"__pycache__", ".pytest_cache", "module.pyc"}
