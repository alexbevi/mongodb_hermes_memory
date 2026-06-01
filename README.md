# MongoDB Memory Provider for Hermes Agent

Hybrid search, time-decay relevance, TTL-driven forgetting, and entity graph traversal — served from a single MongoDB cluster (self-hosted or Atlas free tier).

[![CI](https://github.com/alexbevi/mongodb_hermes_memory/actions/workflows/ci.yml/badge.svg)](https://github.com/alexbevi/mongodb_hermes_memory/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why MongoDB?

Existing Hermes memory providers each pick one storage angle: local SQLite (Holographic), cloud-only API (Mem0, Supermemory, RetainDB), knowledge graph (Hindsight), or vector-only stores. **MongoDB collapses all of them into one backend.**

| Capability | This plugin | Closest alternative |
|---|---|---|
| Native hybrid (vector + BM25) ranking | ✅ Atlas `$rankFusion` | RetainDB ($20/mo cloud-only) |
| Self-hosted *and* cloud with one URI swap | ✅ | OpenViking (AGPL-3.0) |
| TTL-driven automatic forgetting | ✅ Mongo TTL indexes | None |
| Entity graph traversal without a graph DB | ✅ `$graphLookup` | Hindsight (separate KG) |
| Time-decay relevance built into search | ✅ in-pipeline `$exp` | None |
| Free tier with cloud durability | ✅ Atlas free tier | Holographic (local-only) |
| Cross-device sync | ✅ via Atlas | Cloud providers only |

In other words: **vector search, BM25, structured queries, TTL, and graph traversal are all the same database** — no glue between SQLite and a vector store, no separate graph engine.

## Requirements

- Python ≥ 3.10
- A MongoDB 7.0+ cluster, either:
  - [Atlas](https://www.mongodb.com/cloud/atlas) free tier (M0) — recommended; enables `$rankFusion` hybrid search and managed vector indexes
  - Self-hosted (community / enterprise) — works with the portable BM25 + cosine fallback path
- *(Optional)* an embedding provider key — OpenAI or Voyage AI. Without one, the plugin runs as a tuned BM25 + entity-graph store.

## Install

```bash
pip install hermes-mongodb-memory
# Optional vector backends:
pip install "hermes-mongodb-memory[openai]"
pip install "hermes-mongodb-memory[voyage]"
pip install "hermes-mongodb-memory[all]"   # both
```

## Setup

### Path 1 — Guided (recommended)

```bash
hermes memory setup        # pick "mongodb" from the provider list
hermes mongodb init-indexes
```

The wizard prompts for connection URI, database, embedding provider, and tenant scope, then writes:

- `$HERMES_HOME/mongodb.json` — non-secret values
- `$HERMES_HOME/.env` — `HERMES_MONGODB_URI` and any embedding API key

### Path 2 — Manual

```bash
hermes config set memory.provider mongodb
echo 'HERMES_MONGODB_URI="mongodb+srv://USER:PASS@cluster.mongodb.net/?retryWrites=true"' >> ~/.hermes/.env
hermes mongodb init-indexes
```

### Path 3 — Local development

```bash
git clone https://github.com/alexbevi/mongodb_hermes_memory.git
cd mongodb_hermes_memory
pip install -e ".[dev,all]"
# Hermes auto-discovers plugins in ~/.hermes/plugins/<name>/
ln -s "$(pwd)/hermes_mongodb_memory" ~/.hermes/plugins/mongodb
docker compose up -d                            # local Mongo 7 replica set
export HERMES_MONGODB_URI="mongodb://localhost:27017/?replicaSet=rs0"
```

### Atlas free tier (zero-cost path)

1. Create an [Atlas account](https://www.mongodb.com/cloud/atlas) and a free M0 cluster.
2. *Network Access* → Add your IP (or `0.0.0.0/0` for development).
3. *Database Access* → Create a user.
4. *Connect* → Drivers → copy the `mongodb+srv://...` string.
5. Paste it into `hermes memory setup` (or set `HERMES_MONGODB_URI`).
6. *(Optional, enables hybrid search)* Create the search indexes — see [Atlas Search indexes](#atlas-search-indexes) below.

## Configuration

`$HERMES_HOME/mongodb.json` — env vars override file values.

| Key | Default | Description |
|---|---|---|
| `connection_uri` | — | MongoDB URI (env: `HERMES_MONGODB_URI`) |
| `database` | `hermes_memory` | Database name |
| `embedding_provider` | `none` | `none` / `openai` / `voyage` |
| `embedding_api_key` | — | Provider API key (env: `HERMES_MONGODB_EMBEDDING_API_KEY`) |
| `embedding_model` | `text-embedding-3-small` | Embedding model name |
| `embedding_dim` | `1536` | Must match your Atlas Vector Search index |
| `tenant_scope` | `per-workspace` | `global` / `per-workspace` / `per-profile` |
| `auto_extract` | `false` | Run regex extractor at session end |
| `extraction_llm` | `false` | Use Hermes' configured LLM for richer extraction |
| `time_decay_half_life_days` | `30` | Half-life for relevance decay (0 disables) |
| `default_ttl_days` | `0` | Default TTL for new memories (0 = never) |
| `hybrid_vector_weight` | `0.6` | Vector vs BM25 weight (0–1) |
| `prefetch_limit` | `5` | Memories injected into each turn |
| `turn_ttl_days` | `30` | TTL for raw turn capture (0 = forever) |

## Tools

The model can call six tools while the provider is active:

| Tool | Purpose |
|---|---|
| `mongo_remember` | Store a memory with category, entities, tags, optional TTL |
| `mongo_search` | Hybrid query (vector + BM25 + time-decay) — call before answering personal questions |
| `mongo_recall` | Walk the entity co-occurrence graph for related memories |
| `mongo_forget` | Remove or set short TTL on a memory by id |
| `mongo_profile` | Summary of latest preferences, projects, decisions |
| `mongo_reflect` | Cluster recent memories by entity for cross-category synthesis |

## Indexes

`hermes mongodb init-indexes` creates the following on `memories`:

| Name | Shape | Why |
|---|---|---|
| `tenant_category_recent` | `{tenant_id, category, created_at}` | Fast `mongo_profile` and per-category list scans |
| `tenant_entities` | `{tenant_id, entities}` | Entity recall + `$graphLookup` start point |
| `tenant_tags` (sparse) | `{tenant_id, tags}` | Tag filtering without bloating non-tagged docs |
| `ttl_expires_at` (sparse, TTL=0) | `{expires_at}` | MongoDB TTL monitor evicts ephemeral memories |
| `content_text` | `{content: "text"}` | Portable BM25 fallback for self-hosted clusters |

Plus on `turns`:

- `tenant_session_turn` — `{tenant_id, session_id, turn_idx}`
- `turns_ttl` — TTL on `expires_at`

### Atlas Search indexes

For the hybrid-search fast path, create these in the Atlas UI (or via the Atlas Admin API):

```json
// Atlas Search index "hermes_memory_text"
{ "mappings": { "fields": { "content": { "type": "string" } } } }

// Atlas Vector Search index "hermes_memory_vector"
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine" },
    { "type": "filter", "path": "tenant_id" },
    { "type": "filter", "path": "category" },
    { "type": "filter", "path": "entities" }
  ]
}
```

Without these, the plugin transparently falls back to portable BM25 + in-process cosine.

## Testing

```bash
# Unit tests (mongomock-backed, no docker required)
pytest tests/ -v --cov=hermes_mongodb_memory

# Integration tests against a real MongoDB
docker compose up -d
export MONGODB_TEST_URI="mongodb://localhost:27017"
pytest tests/test_integration.py -v -m integration

# Conformance tests against the real Hermes Agent ABC and loader
# (requires Python 3.11+ for hermes-agent)
pip install -e ".[dev,hermes]"
pytest tests/test_hermes_conformance.py -v -m hermes

# Lint
ruff check .
```

## Architecture

```mermaid
flowchart TB
    subgraph Hermes["Hermes Agent (host)"]
        Register["register(ctx)"]
        Provider["MongoDBMemoryProvider"]
        Prefetch["prefetch / queue_prefetch"]
        SyncTurn["sync_turn"]
        Tools["tools (6)"]
        SessionEnd["on_session_end"]

        Searcher["HybridSearcher"]
        Embedder["EmbeddingClient<br/>(openai / voyage / none)"]
        Store["MongoStore"]
        Dispatcher["ToolDispatcher"]
        Regex["RegexExtractor"]
        LLM["LLMExtractor (optional)"]

        Register --> Provider
        Provider --> Prefetch
        Provider --> SyncTurn
        Provider --> Tools
        Provider --> SessionEnd

        Prefetch --> Searcher
        Searcher --> Embedder
        SyncTurn --> Store
        Tools --> Dispatcher
        Dispatcher --> Store
        Dispatcher --> Searcher
        SessionEnd --> Regex
        SessionEnd --> LLM
        Regex --> Store
        LLM --> Store
    end

    subgraph Mongo["MongoDB / Atlas"]
        Memories[("memories")]
        Turns[("turns")]
        Profiles[("profiles")]
        Entities[("entities")]

        RankFusion["$rankFusion<br/>$vectorSearch + $search (BM25)"]
        GraphLookup["$graphLookup<br/>(entity traversal)"]
        TTL["TTL indexes<br/>(automatic forgetting)"]
    end

    Searcher -.->|aggregation| RankFusion
    Store -.->|CRUD + indexes| Memories
    Store -.-> Turns
    Store -.-> Profiles
    Store -.-> Entities
    Dispatcher -.->|mongo_recall| GraphLookup
    Memories -.- TTL

    classDef host fill:#f5f5f5,stroke:#444,color:#222;
    classDef store fill:#e6f4ea,stroke:#2e7d32,color:#1b5e20;
    classDef op fill:#fff3e0,stroke:#ef6c00,color:#5d4037;
    class Register,Provider,Prefetch,SyncTurn,Tools,SessionEnd,Searcher,Embedder,Store,Dispatcher,Regex,LLM host;
    class Memories,Turns,Profiles,Entities store;
    class RankFusion,GraphLookup,TTL op;
```

Background daemon threads handle `prefetch` and `sync_turn` so a flaky cluster never blocks the conversation. A 5-failure / 120-second circuit breaker pauses Mongo writes when something is wrong upstream.

## Releasing

Publishing is scripted in `scripts/publish.sh`:

```bash
scripts/publish.sh build     # build sdist + wheel only (dry run)
scripts/publish.sh test      # build, lint, test, upload to TestPyPI
scripts/publish.sh prod      # same, upload to PyPI, tag v<version>
scripts/publish.sh gh        # push tag and create GitHub release with notes
```

The PyPI targets enforce a clean working tree, assert that `pyproject.toml`
and `hermes_mongodb_memory/__init__.py` agree on the version, run ruff +
pytest, call `python -m build`, and use `twine` to upload. Configure
credentials in `~/.pypirc` or via `TWINE_USERNAME=__token__` /
`TWINE_PASSWORD=pypi-...`.

The `gh` target reads the version, ensures the `v<version>` tag exists
locally and on `origin`, then calls `gh release create` with auto-generated
notes (install snippet, highlights derived from `feat:` / `fix:` commit
subjects since the previous tag, and a compare link). Idempotent — re-runs
detect existing tags and releases and skip safely. Requires `gh` to be
authenticated (`gh auth login`).

## Contributing

PRs welcome. Conventional Commits required (`feat(scope): ...`, `fix(scope): ...`); tests required for new behavior; coverage gate is 80%.

## License

MIT — same as Hermes Agent.
