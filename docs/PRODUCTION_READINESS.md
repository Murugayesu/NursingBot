# Production Readiness Checkup

Audit of the current codebase (commit `6e81344`, "migrate: replace custom
pipeline with LangChain ecosystem") against production-quality bars:
correctness, security, reliability, and operability. Every finding below was
verified by reading the actual source, not inferred.

**Bottom line: not production-ready as-is.** Four issues in the Critical
section will each cause an outage or a data-exposure incident on their own;
none require architectural rework to fix.

---

## Critical

These block a production rollout outright.

### 1. Alembic will crash on every fresh deploy — missing `psycopg2` dependency
`app/config/settings.py:41-44` builds `sync_database_url` with the
`postgresql+psycopg2` driver, and it's what Alembic uses
(`migrations/env.py:15`). Neither `psycopg2` nor `psycopg2-binary` appears
anywhere in `pyproject.toml`'s dependencies — only the async `asyncpg` driver
is declared. `Dockerfile:22` runs `alembic upgrade head` as the very first
thing on container start.

**Effect:** a clean `docker build` + `docker compose up` fails immediately
with `ModuleNotFoundError: No module named 'psycopg2'`, before the API ever
starts. (It happens to work on this machine only because `psycopg2-binary` is
installed globally outside the project's own dependency list.)

**Fix:** add `"psycopg2-binary>=2.9"` to `pyproject.toml` dependencies (or
switch Alembic to `psycopg` v3 and update `sync_database_url`'s driver
string to match).

### 2. Arbitrary file write via upload filename (path traversal)
`app/api/routes/knowledge_bases.py:103-106`:

```python
doc_dir = Path(tempfile.gettempdir()) / "rag_uploads" / str(doc.id)
doc_dir.mkdir(parents=True, exist_ok=True)
file_path = doc_dir / filename          # filename = upload.filename, client-controlled
file_path.write_bytes(content)
```

`filename` comes straight from the multipart upload with no validation.
Two exploitable behaviors:

- **Absolute path override**: `pathlib`'s `/` operator discards the left
  operand entirely when the right side is absolute. A filename of
  `C:\inetpub\wwwroot\evil.dll` (Windows) or `/etc/cron.d/evil` (Linux) makes
  `file_path` equal to that absolute path — `doc_dir` is silently ignored.
- **Relative traversal**: a filename like `..\..\..\some\path\file` walks
  back out of `doc_dir` on write.

There's also no size cap and no extension/content-type allowlist enforced
before the write.

**Fix:** derive the on-disk filename from `doc.id` (already a UUID) plus a
safe, allowlisted extension — never from client input directly. If the
original filename must be preserved, store it only in `original_filename`
(already a separate column) and sanitize with something like
`Path(filename).name` at minimum, rejecting anything that still contains
path separators or resolves outside `doc_dir` after `.resolve()`.

### 3. No authentication or authorization anywhere in the API
Confirmed by grep across `app/api/`: zero references to auth, API keys, JWTs,
or `Authorization` headers. Every route trusts client-supplied identifiers:

- `GET /knowledge-bases?tenant_id=...` (`knowledge_bases.py:48-56`) filters by
  a `tenant_id` the *caller* supplies (defaulting to `"default"`).
- `GET/POST /knowledge-bases/{kb_id}/...` never checks that the caller is
  allowed to see that `kb_id` — it's a bare UUID lookup.

**Effect:** anyone who can reach the API can read, query, and ingest into any
tenant's knowledge base by guessing or enumerating UUIDs, and can pass any
`tenant_id` they like. This isn't a hardening gap, it's the entire
multi-tenancy model resting on an honor system with no enforcement.

**Fix:** require an authenticated principal (API key or OAuth2/JWT) on every
route, and derive `tenant_id` from that principal server-side — never accept
it as a client-supplied query/body field.

### 4. The documented disaster-recovery script is dead code
`scripts/rebuild_qdrant_index.py` (referenced in `README.md` under "Rebuild
Qdrant Index" as *the* recovery path if Qdrant data is lost) imports:

```python
from app.embeddings.providers.factory import get_embedding_provider   # module doesn't exist
from app.storage.qdrant.indexer import ChunkToIndex, QdrantIndexer    # module doesn't exist
```

Both were removed by the "simplify: remove provider abstractions" /
"migrate: replace custom pipeline with LangChain ecosystem" commits. Running
this script today fails immediately with `ModuleNotFoundError`.

**Effect:** if Qdrant data is ever lost or corrupted in production, the
documented recovery procedure does not work.

**Fix:** rewrite it against the current modules —
`app.embeddings.lc_embeddings.get_embeddings()` and
`app.storage.qdrant.vector_store.get_vector_store()` — and add an integration
test that actually runs it end-to-end against a throwaway KB, so this can't
silently rot again.

---

## High

Not immediately fatal, but each one will cause an incident under realistic
production load or operations.

### 5. Migrations baked into container boot + `--reload` in the prod image
`Dockerfile:22`: `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

- With more than one replica, every replica runs `alembic upgrade head`
  concurrently against the same database on every deploy/restart — a race,
  not a controlled release step.
- `--reload` starts a filesystem watcher and reloads workers on file change.
  It's a dev-only flag; it adds overhead and has no place in a production
  image.

**Fix:** run `alembic upgrade head` once as a standalone release step (CI job
or init container), and use a plain `uvicorn app.main:app --host 0.0.0.0
--port 8000` (or Gunicorn with `UvicornWorker`, sized to CPU count) for the
running container. Details and rationale: [`DATABASE_SETUP.md` §2.7](DATABASE_SETUP.md#27-migrations--run-them-as-a-release-step-not-a-boot-step).

### 6. CORS: `allow_origins=["*"]` + `allow_credentials=True`
`app/main.py:57-63`. This exact combination is both spec-invalid (browsers
refuse credentialed requests to a wildcard origin) and, independent of that,
`allow_origins=["*"]` alone is unsafe for a production API that isn't
intentionally fully public.

**Fix:** set explicit allowed origins from an environment variable per
deployment, and only set `allow_credentials=True` if you actually rely on
cookie-based auth.

### 7. Docker image runs as root
`Dockerfile` has no `USER` directive — the app process runs as root inside
the container. **Fix:** create a non-root user in the image and switch to it
before `CMD`.

### 8. No `.dockerignore` — secrets and bloat can end up in the image
`Dockerfile:19` does `COPY . .` with nothing excluding `.env`, `.git`,
`tests/`, or caches from the build context. If `.env` exists in the build
directory at build time (the normal local workflow, per the README's own
Quick Start), it gets baked into an image layer.

**Fix:** add a `.dockerignore` mirroring `.gitignore` plus `.git/`, `.env`,
`tests/`.

### 9. `slowapi` is a dependency that does nothing
`pyproject.toml` declares `slowapi>=0.1.9`, but grep across `app/` finds no
`Limiter`, no `slowapi` import, no rate-limit decorator anywhere. There is
**no rate limiting** on any endpoint — including unauthenticated file upload
and the `/query` endpoint, which triggers local embedding inference,
reranking, and an LLM call per request.

**Fix:** either wire up `slowapi` (or push rate limiting to a reverse
proxy/API gateway) before production traffic, or drop the unused dependency.
Given finding #3, this is also your only current defense against one caller
exhausting the service for everyone.

### 10. Sparse embedding model reloaded on every single query
`app/retrieval/retriever.py:48-57` — `SparseQdrantRetriever._aget_relevant_documents`
instantiates `SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")`
fresh on every call. Contrast with the dense embedder
(`app/embeddings/lc_embeddings.py:34-43`, module-level cache) and the
reranker (`app/reranking/jina.py:22-27`, `_model_cache` dict) — both
correctly cache their models.

**Effect:** every hybrid-search query pays model-load latency and memory
churn that should be a one-time cost.

**Fix:** cache the `SparseTextEmbedding` instance the same way the other two
loaders do (module-level dict keyed by model name).

### 11. Synchronous RAGAS evaluation blocks the event loop
`app/evaluation/ragas_runner.py:130-139` calls `ragas.evaluate(...)` — a
synchronous, CPU-heavy call — directly inside `async def run(...)`, with no
`asyncio.to_thread`/executor offload. This runs via FastAPI
`BackgroundTasks` on the same event loop that serves every other request.

**Effect:** while an evaluation run is in progress, `/health`, `/query`, and
every other concurrent request stall behind it.

**Fix:** `await asyncio.to_thread(evaluate, dataset, metrics=[...])`.

### 12. Health check swallows and never logs failures, and can hang
`app/api/routes/health.py:15-21`:

```python
try:
    async with engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    pg_ok = True
except Exception as e:
    pg_ok = False        # `e` is never used or logged
```

Two problems: the actual exception (connection refused vs. auth failure vs.
timeout — very different operational meanings) is discarded, so an operator
sees only `"postgres": "unreachable"` in the response with nothing in the
logs to diagnose from. There's also no timeout on either the Postgres or
Qdrant check, so a hung connection hangs `/health` itself — which is exactly
the endpoint an orchestrator's liveness/readiness probe depends on.

**Fix:** `logger.warning("postgres health check failed", error=str(e))`
(structlog is already set up and used elsewhere in this codebase), and wrap
both checks in `asyncio.wait_for(..., timeout=2)`.

---

## Medium

Worth fixing before go-live; won't necessarily cause an immediate incident.

### 13. Databases are published on host ports by default
`docker-compose.yml:12-13,30-32` publish Postgres (5432) and Qdrant
(6333/6334) to the host's network interfaces. Fine for local dev; if this
compose file is reused as-is on a production host, both databases become
reachable from outside the Docker network. See
[`DATABASE_SETUP.md` §2.3/§3.3](DATABASE_SETUP.md#23-network--auth-hardening)
for the production topology — don't publish these ports at all in prod, or
bind them to `127.0.0.1`.

### 14. Weak/empty default secrets, nothing enforces overriding them
`docker-compose.yml`: `POSTGRES_PASSWORD:-ragpassword`, `QDRANT_API_KEY:-`
(blank = no auth). Reasonable dev defaults, but nothing in the app refuses to
boot with these in `APP_ENV=production`.

**Fix:** add a startup assertion in `app/config/settings.py` (or
`app/main.py`'s lifespan) that raises if `app_env == "production"` and
`postgres_password` or `qdrant_api_key` matches a known dev default / is
empty. Also add a `.env.production.example` documenting required overrides.

### 15. No CI pipeline
No `.github/workflows` (or any CI config) exists. `ruff`, `mypy`, and
`pytest` are all declared as dev dependencies (`pyproject.toml`
`[project.optional-dependencies].dev`) but nothing runs them automatically on
a PR or push.

**Fix:** add a workflow running `ruff check .`, `mypy app`, and
`pytest tests/unit` on every PR at minimum; add `pytest tests/integration`
against ephemeral Postgres+Qdrant service containers if you want the
multi-tenancy and ingestion paths covered before merge.

### 16. Upload size/type is trusted, not enforced
`app/api/routes/knowledge_bases.py` `upload_documents` accepts
`upload.content_type` as given by the client with no sniffing, and there's no
per-file or per-request size cap. `mime_type` then selects the LangChain
loader (`app/ingestion/loaders/lc_loaders.py:23-32`) — an unrecognized or
mislabeled MIME type silently falls back to `TextLoader` instead of being
rejected.

**Fix:** sniff content type server-side (e.g. `python-magic` or the first
bytes), enforce an explicit allowlist, and cap upload size both at the
application layer and at the reverse proxy.

### 17. Embedding dimension hardcoded in three places instead of derived from config
`app/storage/qdrant/collections.py:27` (`ensure_collection(..., dense_dim: int
= 1024)`), `app/embeddings/lc_embeddings.py:23` (`DIMENSION = 1024`), and
`app/ingestion/pipeline.py:164` (`embedding_dimension=1024`) all hardcode
1024 (BGE-M3's output size). But `RAGConfig.embedding.model`
(`app/config/rag_config.py:17-19`) is presented as a per-knowledge-base
configurable field. The moment any KB is configured with a model whose output
isn't 1024-dim, ingestion will fail with a Qdrant vector-size mismatch — the
"configurable embedding" feature doesn't actually work end-to-end.

**Fix:** derive the dimension from the loaded embedding model
(`len(embeddings.embed_query("x"))` once at load time, or a lookup table) and
thread it through instead of a hardcoded constant.

### 18. No global exception handler / request correlation
An unhandled exception in any route falls through to FastAPI's default
handler. There's no request-ID middleware binding a correlation ID into
`structlog`'s context, making it harder to correlate a single user-facing
error across the ingestion pipeline's background task, the query path, and
logs.

**Fix:** add middleware that generates/propagates a request ID and binds it
via `structlog.contextvars`, plus a global exception handler that logs with
that ID and returns a consistent error envelope.

---

## Low

Nice-to-haves; not blockers.

### 19. `extra="ignore"` on Settings hides typo'd env vars
`app/config/settings.py:15` — a mistyped env var (e.g.
`POSTGRES_PASSWROD`) is silently ignored, and the app falls back to the
hardcoded default rather than failing to start. Consider `extra="forbid"` in
production to catch misconfiguration at boot instead of at query time.

### 20. Single `/health` conflates liveness and readiness
Fine for a small deployment; if this moves to Kubernetes, a transient Qdrant
blip will make `/health` report `"degraded"` and can cause an orchestrator to
restart an otherwise-healthy API process. Consider splitting into
`/health/live` (process up) and `/health/ready` (dependencies up) if/when you
deploy to k8s.

### 21. No `HEALTHCHECK` in the Dockerfile
`docker-compose.yml` defines healthchecks for `postgres` and `qdrant` but not
for `api`. Add a `HEALTHCHECK` instruction hitting `/health` so Docker itself
can detect a hung API process without external tooling.

---

## What's already solid

Worth naming, since a checkup that's all findings is misleading:

- Async SQLAlchemy engine with `pool_pre_ping=True` (`database.py:13`) — avoids serving requests on dead connections.
- Cascading foreign keys throughout the schema (`ON DELETE CASCADE`) — deleting a KB cleanly removes its documents/chunks/jobs.
- Content-hash-based deduplication in the ingestion pipeline (`pipeline.py:136-147`) avoids redundant re-embedding of unchanged documents.
- Reranker and dense-embedding model loading are both properly cached as singletons and the reranker's async path correctly offloads to a thread executor (`jina.py:94-106`) — the sparse retriever (#10 above) is the one path that doesn't follow this pattern.
- Structured logging via `structlog` is consistently used across the ingestion and query paths.
- Qdrant retrieval filters *do* scope by `tenant_id` + `knowledge_base_id` (`retriever.py:158-165`) — real defense in depth at the vector-store layer, it's just undermined by having no authentication in front of it (#3).
- Langfuse tracing fails soft (`langfuse_tracer.py:40-53`) — a misconfigured or unreachable Langfuse instance degrades observability, not availability.
- Migrations are properly version-controlled via Alembic with `compare_type=True`.

---

## Go-live checklist

- [ ] Fix #1 (add `psycopg2-binary` dependency) — nothing else works until this is fixed
- [ ] Fix #2 (sanitize upload filenames)
- [ ] Add authentication + tenant derivation from the authenticated principal (#3)
- [ ] Fix or replace the DR script (#4), then actually test a restore
- [ ] Split migrations out of the container boot command; drop `--reload` (#5)
- [ ] Lock down CORS origins (#6)
- [ ] Non-root Docker user (#7) + `.dockerignore` (#8)
- [ ] Wire up rate limiting or remove `slowapi` (#9)
- [ ] Cache the sparse embedding model (#10)
- [ ] Offload `ragas.evaluate` to a thread (#11)
- [ ] Log health-check failures + add timeouts (#12)
- [ ] Don't publish DB ports on a production host (#13)
- [ ] Enforce non-default secrets in `APP_ENV=production` (#14)
- [ ] Stand up CI running ruff/mypy/pytest (#15)
- [ ] Enforce upload size/type limits (#16)
- [ ] Derive embedding dimension from the configured model (#17)
- [ ] Everything in [`DATABASE_SETUP.md`](DATABASE_SETUP.md) §5's bring-up runbook (pooling, TLS, backups, monitoring)
