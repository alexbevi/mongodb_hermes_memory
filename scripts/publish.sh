#!/usr/bin/env bash
# Build and upload hermes-mongodb-memory to PyPI (or TestPyPI), and
# create the matching GitHub release.
#
# Usage:
#   scripts/publish.sh           # uploads to TestPyPI
#   scripts/publish.sh prod      # uploads to PyPI + tags v<version>
#   scripts/publish.sh build     # build only — no upload
#   scripts/publish.sh gh        # push tag + create GitHub release with notes
#
# Typical flow:
#   scripts/publish.sh test      # validate on TestPyPI
#   scripts/publish.sh prod      # ship to PyPI, creates local tag
#   scripts/publish.sh gh        # publish GitHub release from that tag
#
# Credentials: configure ~/.pypirc with [pypi] and [testpypi] sections, or set
# TWINE_USERNAME=__token__ and TWINE_PASSWORD=pypi-<token> in the environment.
# GitHub release uses ``gh`` (must be authenticated: gh auth login).
#
# Requirements: python (3.10+), build, twine, git, gh (for the gh target).
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

# --- gh target: short-circuit before build/test ------------------------

if [[ "$TARGET" == "gh" ]]; then
    if ! command -v gh >/dev/null 2>&1; then
        echo "error: gh CLI not found. Install: brew install gh" >&2
        exit 1
    fi
    if ! gh auth status >/dev/null 2>&1; then
        echo "error: gh not authenticated. Run: gh auth login" >&2
        exit 1
    fi

    TAG="v$VERSION"

    # Ensure the tag exists locally; create it if not.
    if ! git rev-parse "$TAG" >/dev/null 2>&1; then
        echo "→ tag $TAG missing locally; creating from HEAD"
        git tag -a "$TAG" -m "Release $TAG"
    fi

    # Push the tag (idempotent — git push silently no-ops if remote already has it).
    if ! git ls-remote --tags origin "$TAG" | grep -q "$TAG"; then
        echo "→ pushing tag $TAG to origin"
        git push origin "$TAG"
    else
        echo "→ tag $TAG already on origin"
    fi

    # Skip if the release already exists.
    if gh release view "$TAG" >/dev/null 2>&1; then
        echo "→ GitHub release $TAG already exists; skipping create"
        gh release view "$TAG" --web 2>/dev/null || gh release view "$TAG"
        exit 0
    fi

    # Build release notes: highlights + auto-generated diff section.
    PREV_TAG="$(git tag --list 'v*' --sort=-version:refname | grep -v "^${TAG}$" | head -n1 || true)"
    NOTES_FILE="$(mktemp -t release-notes.XXXXXX)"
    trap 'rm -f "$NOTES_FILE"' EXIT

    {
        echo "## Install"
        echo
        echo '```bash'
        echo "pip install hermes-mongodb-memory==$VERSION"
        echo '```'
        echo
        echo "## Highlights"
        echo
        # Pull feat/fix subjects since last tag for the highlights bullet list.
        if [[ -n "$PREV_TAG" ]]; then
            git log --no-merges --pretty=format:'- %s' "${PREV_TAG}..${TAG}" \
                | grep -E '^- (feat|fix|perf|refactor|docs)' \
                | sed -E 's|^- ([a-z]+)(\([^)]+\))?: |- **\1\2:** |' \
                || echo "- See full changelog below"
        else
            echo "- Initial release"
        fi
        echo
        echo "## What's Changed"
        echo
        if [[ -n "$PREV_TAG" ]]; then
            echo "Full diff: [\`${PREV_TAG}...${TAG}\`](https://github.com/alexbevi/mongodb_hermes_memory/compare/${PREV_TAG}...${TAG})"
        else
            echo "Initial release."
        fi
    } > "$NOTES_FILE"

    echo "→ creating GitHub release $TAG"
    gh release create "$TAG" \
        --title "$TAG" \
        --notes-file "$NOTES_FILE" \
        --verify-tag

    URL="$(gh release view "$TAG" --json url --jq '.url')"
    echo
    echo "→ release published: $URL"
    exit 0
fi

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
        echo "error: unknown target '$TARGET' (expected: build, test, prod, gh)" >&2
        exit 2
        ;;
esac
