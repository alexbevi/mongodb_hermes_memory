# MongoDB Memory Provider for Hermes Agent

Hybrid search, time-decay relevance, TTL-driven forgetting, and entity graph traversal — all served from a single MongoDB cluster (self-hosted or Atlas free tier).

> Status: scaffolding. Full README arrives with the final commit; this file is replaced before v0.1.0 ships.

## Why MongoDB?

- **One backend, every memory shape.** Vector search, BM25, structured queries, TTL, and graph traversal in the same database — no glue between SQLite and a vector store.
- **Native hybrid search** via Atlas `$rankFusion`. On self-hosted MongoDB the plugin falls back to `$text` + cosine-similarity merge so it works anywhere.
- **Self-hosted or Atlas** with one URI change. Atlas free tier keeps onboarding zero-cost.

## Install

```bash
pip install hermes-mongodb-memory
```

Then run `hermes memory setup` and pick `mongodb`. See the full installation guide in the final README.

## License

MIT
