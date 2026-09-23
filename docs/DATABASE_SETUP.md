# Database Server Setup Guide — PostgreSQL + Qdrant

This platform depends on two stateful services:

| Store | Role | Data it holds |
|---|---|---|
| **PostgreSQL 16** | System of record | Knowledge bases, documents, document versions/raw+cleaned text, chunk records, ingestion jobs, evaluation datasets/runs |
| **Qdrant v1.12** | Vector index | Dense (BGE-M3, 1024-dim) + sparse (SPLADE) vectors per knowledge base, one collection per KB (`kb_<uuid hex>`) |

Postgres is the source of truth; Qdrant is a rebuildable index derived from it
(`scripts/rebuild_qdrant_index.py` — **currently broken**, see
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md#critical) before you rely on it).
Losing Qdrant is recoverable in principle; losing Postgres is not.

---

## 1. Local / dev (already wired up)

```bash
cp .env.example .env
docker compose up -d
docker compose exec api alembic upgrade head
```

This is `docker-compose.yml`'s `postgres` + `qdrant` services — fine for a laptop,
**not** what you point at production. The rest of this guide is the production path.

---

## 2. PostgreSQL — production setup

### 2.1 Choosing a deployment

| Option | When |
|---|---|
| Managed (RDS / Cloud SQL / Azure Database for PostgreSQL) | Default recommendation — offloads patching, backups, failover |
| Self-hosted VM | Only if compliance/cost forces it; everything below assumes this path |
| The `postgres` container from `docker-compose.yml` | **Dev only.** No backup, no TLS, weak default password, port published to the host |

### 2.2 Self-hosted install (Ubuntu 22.04/24.04 example)

```bash
sudo apt update
sudo apt install -y postgresql-16 postgresql-contrib-16

sudo -u postgres psql <<'SQL'
CREATE ROLE rag_app WITH LOGIN PASSWORD 'CHANGE_ME_LONG_RANDOM';
CREATE DATABASE ragdb OWNER rag_app;
-- least privilege: app role only, never a superuser
REVOKE ALL ON DATABASE ragdb FROM PUBLIC;
GRANT CONNECT ON DATABASE ragdb TO rag_app;
SQL
```

Match the version to `postgres:16-alpine` used in `docker-compose.yml` so behavior
is identical between dev and prod.

### 2.3 Network & auth hardening

- `listen_addresses` in `postgresql.conf`: bind only to the app subnet's interface, never `*` on a box with a public IP.
- `pg_hba.conf`: replace any `trust`/`md5` entries with `scram-sha-256`, scoped to the application's private subnet/security group only.
- Require TLS in transit: `ssl = on` in `postgresql.conf`, and set the app's `postgres_host`/connection string to use `sslmode=require` (or `verify-full` with a CA cert pinned) — the current `database_url`/`sync_database_url` in `app/config/settings.py:32-44` don't set `sslmode` at all, so add it via a connection URL query param or `connect_args` once TLS is enabled server-side.
- Never expose 5432 to the public internet. In production, do **not** reuse `docker-compose.yml`'s `ports: ["${POSTGRES_PORT:-5432}:5432"]` — that binds to all interfaces on the host by default.

### 2.4 Connection pooling — do the math before you deploy

The app's engine (`app/storage/postgres/database.py:11-17`) opens
`pool_size=10, max_overflow=20` = **up to 30 connections per app replica**.
Postgres' default `max_connections` is 100.

> 3 app replicas × 30 = 90 connections, leaving almost no headroom for `psql`,
> migrations, or a second service. **Put PgBouncer (transaction pooling mode)
> in front of Postgres** before you scale past a single replica, and point
> `POSTGRES_HOST`/`POSTGRES_PORT` at PgBouncer instead of Postgres directly.

### 2.5 Backups & recovery

- **Logical backups**: nightly `pg_dump -Fc ragdb` to object storage, retained ≥ 14 days.
- **Point-in-time recovery**: enable WAL archiving (`archive_mode = on`) and use `pgBackRest` or `wal-g` for continuous base backups + WAL shipping if RPO < 24h matters.
- **Test restores on a schedule** — an untested backup is not a backup. Restore into a scratch instance monthly and run `alembic current` against it to confirm the schema matches.
- Document sizing: `document_versions.raw_text`/`cleaned_text` store full document text (TOAST-compressed automatically by Postgres, but still the largest table by volume). Provision disk at 3–5× your expected corpus size to leave room for growth, WAL, and index bloat.

### 2.6 Monitoring

- Enable `pg_stat_statements` for slow-query visibility.
- Run `postgres_exporter` → Prometheus/Grafana. Alert on: connection count vs `max_connections`, replication lag (if HA), disk usage, long-running transactions.

### 2.7 Migrations — run them as a release step, not a boot step

`Dockerfile:22` currently runs `alembic upgrade head && uvicorn ...` as the
container's `CMD`. That's convenient for a single dev container but **breaks
with more than one replica** — every replica races to acquire Alembic's
migration lock and apply the same revision concurrently. In production:

```bash
# One-off job/init-container/CI release step, run exactly once per deploy:
alembic upgrade head

# App containers then start with a plain runtime command (no migration, no --reload):
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers <N>
```

(See [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md#high) for the matching
Dockerfile finding — this also currently fails outright due to a missing driver
dependency; fix that first.)

---

## 3. Qdrant — production setup

### 3.1 Choosing a deployment

| Option | When |
|---|---|
| Qdrant Cloud (managed) | Default recommendation for most teams |
| Self-hosted cluster (Kubernetes + Helm chart, distributed mode) | High volume / data residency requirements |
| Single `qdrant` container | Dev only, or a very small single-tenant deployment with accepted downtime risk |

### 3.2 Sizing — do this before provisioning

Dense vectors are BGE-M3 at 1024 dims, float32 → **4 KB/vector raw**. With HNSW
graph overhead (`m=16, ef_construct=100`, set in `app/storage/qdrant/collections.py:45`)
budget **~1.5–2× raw size in RAM** per collection unless you enable on-disk
storage for vectors (`on_disk=True` in `VectorParams`, trading latency for RAM).

```
RAM budget ≈ vector_count × 4KB × 1.8   (dense, in-memory HNSW)
           + sparse index overhead (small; sparse index already configured on_disk=False)
```

Example: 5M chunks across all KBs ≈ 5,000,000 × 4KB × 1.8 ≈ **36 GB RAM** just for
dense vectors, before payload/metadata. Re-run this per KB since each KB gets
its own collection (`collection_name()` in `collections.py:19-21`) — many
small KBs is fine; a handful of huge ones needs real planning.

### 3.3 Security hardening

- **Set `QDRANT__SERVICE__API_KEY`** — it defaults to blank in both
  `docker-compose.yml:28` and `.env.example:23`, meaning Qdrant runs with
  **no authentication** unless you explicitly set it. Treat an empty API key
  as a hard blocker for any non-local deployment.
- Put TLS in front of Qdrant (its own TLS config, or a reverse proxy/load
  balancer terminating TLS) — REST (6333) and gRPC (6334) are both plaintext
  by default.
- Don't publish 6333/6334 to a public interface; keep Qdrant on the private
  network the API service lives on, same as Postgres.

### 3.4 Backups

- Use Qdrant's native snapshot API per collection: `POST /collections/{name}/snapshots`, then ship the snapshot file to object storage. Automate this on a schedule (cron/sidecar) — there is nothing in this repo that does it today.
- Keep `scripts/rebuild_qdrant_index.py` as your last-resort DR path from Postgres — but **it is currently broken** (imports modules removed in the LangChain migration). Fix it before you depend on it; see the checkup doc.

### 3.5 Scaling / HA

- Distributed mode with `replication_factor ≥ 2` if you need to survive a node loss.
- Shard large collections (millions of vectors in a single KB) across nodes.
- Set container/pod memory limits matching §3.2's sizing — Qdrant will OOM rather than gracefully degrade if you undersize RAM for in-memory HNSW.

### 3.6 Monitoring

Qdrant exposes a Prometheus `/metrics` endpoint — scrape it and alert on memory
usage approaching the limit computed in §3.2, and on disk usage for snapshot
storage.

---

## 4. Secrets & environment

Every default in `docker-compose.yml` and `.env.example` is a **dev convenience,
not a production value**:

| Variable | Dev default | Production requirement |
|---|---|---|
| `POSTGRES_PASSWORD` | `ragpassword` | Long random secret, injected via secret manager (never committed) |
| `QDRANT_API_KEY` | *(blank — no auth)* | Required, long random secret |
| `OPENAI_API_KEY` | placeholder | Real key, injected via secret manager |
| `LANGFUSE_SECRET_KEY` | *(blank)* | Set if `LANGFUSE_ENABLED=true` |

Nothing in the app currently enforces this (see
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md#medium) — no startup check
rejects default/empty secrets when `APP_ENV=production`). Until that's added,
treat this table as a manual pre-flight checklist.

Use your platform's secret manager (AWS Secrets Manager, GCP Secret Manager,
Vault, sealed-secrets in k8s, …) rather than a `.env` file on the box for
anything beyond a single-VM deployment.

---

## 5. Production bring-up runbook

1. Provision Postgres 16 (managed or §2.2) — dedicated `rag_app` role, TLS on, network-restricted.
2. Provision Qdrant (managed or §3.1) — API key set, TLS in front, network-restricted.
3. Put PgBouncer in front of Postgres if running >1 API replica (§2.4).
4. Generate and store all secrets from §4 in your secret manager.
5. Run `alembic upgrade head` once, as a standalone release step (§2.7) — **not** via the default Dockerfile `CMD`.
6. Deploy the API pointed at the pooled Postgres endpoint and the Qdrant endpoint, `APP_ENV=production`, `LOG_LEVEL=INFO`, CORS origins locked to your real frontend domain(s).
7. Confirm `/health` reports `"status": "ok"` for both services.
8. Wire up automated backups: nightly `pg_dump`/WAL archiving (§2.5) and scheduled Qdrant snapshots (§3.4).
9. Wire up monitoring/alerting (§2.6, §3.6) before onboarding real traffic.
10. Do a full restore drill (Postgres backup → scratch instance; Qdrant snapshot → scratch collection) before go-live, not after an incident.

---

## 6. Environment variable reference

Full list with descriptions: [`.env.example`](../.env.example). The
database-relevant subset:

```
POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
QDRANT_HOST / QDRANT_HTTP_PORT / QDRANT_GRPC_PORT / QDRANT_API_KEY
```

For the full list of correctness/security issues found while writing this
guide (several of which block a safe production rollout as-is), see
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).
