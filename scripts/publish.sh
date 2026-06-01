#!/usr/bin/env bash
# Build and upload hermes-mongodb-memory to PyPI (or TestPyPI).
#
# Usage:
#   scripts/publish.sh           # uploads to TestPyPI
#   scripts/publish.sh prod      # uploads to PyPI
#   scripts/publish.sh build     # build only — no upload
#
# Credentials: configure ~/.pypirc with [pypi] and [testpypi] sections, or set
# TWINE_USERNAME=__token__ and TWINE_PASSWORD=pypi-<token> in the environment.
#
# Requirements: python (3.10+), build, twine, git.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-test}"

PYTHON="${PYTHON:-python3}"

# --- 1. Sanity checks ----------------------------------------------------

if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: working tree is dirty — commit or stash first." >&2
    git status --short >&2
    exit 1
fi

VERSION_PYPROJECT="$($PYTHON -c "
import re, pathlib
m = re.search(r'^version\s*=\s*\"([^\"]+)\"', pathlib.Path('pyproject.toml').read_text(), re.MULTILINE)
print(m.group(1))
")"
VERSION_INIT="$($PYTHON -c "
import re, pathlib
m = re.search(r'__version__\s*=\s*\"([^\"]+)\"', pathlib.Path('hermes_mongodb_memory/__init__.py').read_text())
print(m.group(1))
")"

if [[ "$VERSION_PYPROJECT" != "$VERSION_INIT" ]]; then
    echo "error: version mismatch — pyproject.toml=$VERSION_PYPROJECT, __init__.py=$VERSION_INIT" >&2
    exit 1
fi
VERSION="$VERSION_PYPROJECT"
echo "→ version: $VERSION"

# --- 2. Install build tooling -------------------------------------------

$PYTHON -m pip install --quiet --upgrade build twine

# --- 3. Run tests + lint before shipping --------------------------------

if [[ -d ".venv" ]]; then
    PY=".venv/bin/python"
else
    PY="$PYTHON"
fi

echo "→ running ruff"
$PY -m ruff check .

echo "→ running tests"
$PY -m pytest tests/ -q --cov=hermes_mongodb_memory --cov-report=term

# --- 4. Clean and build -------------------------------------------------

echo "→ cleaning previous artifacts"
rm -rf dist/ build/ ./*.egg-info hermes_mongodb_memory.egg-info

echo "→ building sdist + wheel"
$PYTHON -m build

echo "→ verifying distribution metadata"
$PYTHON -m twine check dist/*

ls -lh dist/

# --- 5. Upload (or stop after build) ------------------------------------

case "$TARGET" in
    build)
        echo "→ build complete; skipping upload (TARGET=build)"
        exit 0
        ;;
    test|testpypi)
        echo "→ uploading to TestPyPI"
        $PYTHON -m twine upload --repository testpypi dist/*
        echo
        echo "Test install:"
        echo "  pip install --index-url https://test.pypi.org/simple/ \\"
        echo "    --extra-index-url https://pypi.org/simple/ hermes-mongodb-memory==$VERSION"
        ;;
    prod|pypi)
        echo "→ uploading to PyPI"
        read -r -p "About to publish $VERSION to production PyPI. Continue? [y/N] " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo "aborted."
            exit 1
        fi
        $PYTHON -m twine upload dist/*

        # Tag the release on success.
        if ! git rev-parse "v$VERSION" >/dev/null 2>&1; then
            git tag -a "v$VERSION" -m "Release v$VERSION"
            echo "→ created tag v$VERSION (push with: git push origin v$VERSION)"
        fi
        echo
        echo "Install:"
        echo "  pip install hermes-mongodb-memory==$VERSION"
        ;;
    *)
        echo "error: unknown target '$TARGET' (expected: build, test, prod)" >&2
        exit 2
        ;;
esac
