# Project Workflow

Change → branch → PR → CI → merge → tag → image → migrate → deploy.

## 1. Local development

```bash
make venv                    # once; a venv stops other editable installs shadowing `app`
cp .env.example .env         # set OPENAI_API_KEY at minimum
make up                      # postgres + qdrant + api (auto-reload, runs migrations)
```

The compose `api` service overrides the image command to migrate and `--reload`.
The image itself does neither (production behaviour).

Daily loop: edit, then `make check` (ruff, mypy, unit tests). `make fmt` auto-fixes.
`make test-int` needs Postgres and Qdrant running.

## 2. Schema changes

1. Edit the model in `app/models/`.
2. `make revision m="describe change"`, then review the generated file by hand.
3. `alembic upgrade head` on a fresh DB, and `alembic downgrade -1` to confirm it reverses.
4. Commit the model and migration together.

Qdrant collections are created lazily per knowledge base
(`app/storage/qdrant/collections.py`); changing vector size or config means
re-ingesting or rebuilding the affected collections.

## 3. Branching and PRs

- Branch from `main` as `feat/...`, `fix/...`, `chore/...`.
- Open a PR; the template lists what to confirm.
- CI (`.github/workflows/ci.yml`) must be green before merge:

| Job | Gate |
|---|---|
| Lint & type-check | `ruff check`, `ruff format --check`, `mypy -p app` |
| Unit tests | `pytest tests/unit` (no services) |
| Integration tests | `alembic upgrade head`, then `pytest tests/integration` against Postgres + Qdrant service containers |
| Docker image builds | Dockerfile builds (no push) |

Dependabot opens weekly PRs for pip, Docker and Actions updates.

## 4. Release

Tag a commit on the default branch:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

`release.yml` builds and pushes `ghcr.io/<owner>/<repo>:0.2.0` and `:sha-<short>`.

## 5. Deploy

Order matters and is manual on purpose:

1. **Back up** Postgres (see `DATABASE_SETUP.md` §2.5).
2. **Migrate once**, using the new image, before any new app container starts:
   ```bash
   docker run --rm --env-file prod.env ghcr.io/<owner>/<repo>:0.2.0 alembic upgrade head
   ```
3. **Roll out** the new image (one replica at a time if possible).
4. **Verify** `GET /health` reports `ok` for postgres and qdrant, then run one query.
5. **Roll back** by redeploying the previous tag. Migrations must stay
   backward-compatible with the previous release (add columns nullable, drop
   only in a later release) so this stays safe.

Pre-flight for any production deploy: [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)
go-live checklist and [`DATABASE_SETUP.md`](DATABASE_SETUP.md) §5.

## 6. Runtime data flow (for reference)

```
Upload → Postgres (status: UPLOADED)
POST /ingest → background job: load → split → embed (BGE-M3) → Qdrant (dense+sparse) → chunk rows in Postgres
POST /query  → embed → dense + sparse retrieval → RRF → Jina rerank → LLM → answer + sources (Langfuse trace)
POST /evaluate → per-question retrieval+generation → RAGAS metrics → stored in evaluation_runs
```
