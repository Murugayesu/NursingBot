# RAG Platform V1

A production-oriented, **configuration-driven** Retrieval-Augmented Generation platform.
Swap data + configuration to get Legal RAG, Medical RAG, Customer Support RAG — without rewriting the pipeline.

---

## Architecture

```
Upload → Postgres → Clean → Chunk → Embed (BGE-M3) → Qdrant (dense + sparse)
                                                              ↓
Query → Embed → Hybrid Retrieval → RRF → Jina Reranker → Context → LLM → Answer + Sources
                                                              ↓
                                                         Langfuse Trace
                                                              ↓
                                                        RAGAS Evaluation
```

---

## Stack 

| Layer | Technology |
|---|---|
| API | FastAPI |
| Database | PostgreSQL (SQLAlchemy async) |
| Migrations | Alembic |
| Vector DB | Qdrant (dense + sparse named vectors) |
| Embeddings | BGE-M3 local (FlagEmbedding) |
| Sparse vectors | SPLADE via fastembed |
| Reranker | Jina Reranker v2 local (transformers) |
| LLM | OpenAI-compatible API |
| Observability | Langfuse |
| Evaluation | RAGAS |
| Async jobs | FastAPI BackgroundTasks |
| Containerisation | Docker Compose |

---

## Quick Start

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and OPENAI_BASE_URL at minimum
```

### 2. Start services

```bash
docker compose up -d
```

Services: `postgres:5432`, `qdrant:6333`, `api:8000`

### 3. Run migrations (first time)

```bash
docker compose exec api alembic upgrade head
```

### 4. Open API docs

```
http://localhost:8000/docs
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/knowledge-bases` | Create a knowledge base |
| `GET` | `/knowledge-bases` | List knowledge bases |
| `GET` | `/knowledge-bases/{id}` | Get a knowledge base |
| `POST` | `/knowledge-bases/{id}/documents` | Upload documents (multipart) |
| `GET` | `/knowledge-bases/{id}/documents` | List documents |
| `POST` | `/knowledge-bases/{id}/ingest` | Trigger async ingestion |
| `GET` | `/ingestion-jobs/{id}` | Poll ingestion job status |
| `POST` | `/knowledge-bases/{id}/query` | Query the knowledge base |
| `POST` | `/knowledge-bases/{id}/datasets` | Create evaluation dataset |
| `POST` | `/knowledge-bases/{id}/evaluate` | Run RAGAS evaluation |
| `GET` | `/knowledge-bases/{id}/evaluation-runs/{run_id}` | Poll evaluation run |

---

## End-to-End Example

```bash
# 1. Create a knowledge base
curl -X POST http://localhost:8000/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs", "tenant_id": "default"}'
# → {"id": "<kb_id>", ...}

# 2. Upload documents
curl -X POST http://localhost:8000/knowledge-bases/<kb_id>/documents \
  -F "files=@/path/to/document.pdf"

# 3. Trigger ingestion
curl -X POST http://localhost:8000/knowledge-bases/<kb_id>/ingest
# → {"id": "<job_id>", "status": "PENDING"}

# 4. Poll until COMPLETED
curl http://localhost:8000/ingestion-jobs/<job_id>

# 5. Query
curl -X POST http://localhost:8000/knowledge-bases/<kb_id>/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}'
```

---

## RAG Configuration

Each knowledge base stores a `rag_config` JSON blob that controls every pipeline stage:

```json
{
  "chunking": {"strategy": "recursive", "chunk_size": 800, "chunk_overlap": 100},
  "embedding": {"provider": "bge_m3", "model": "BAAI/bge-m3", "device": "cpu"},
  "retrieval": {"dense": true, "sparse": true, "dense_top_k": 20, "sparse_top_k": 20, "fusion": "rrf"},
  "reranking": {"enabled": true, "provider": "jina", "model": "jinaai/jina-reranker-v2-base-multilingual", "top_k": 5},
  "generation": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 2048},
  "prompt": {"version": "qa_v1"}
}
```

---

## Document Lifecycle

```
UPLOADED → EXTRACTING → EXTRACTED → CLEANING → CLEANED
        → CHUNKING → CHUNKED → EMBEDDING → EMBEDDED → INDEXING → INDEXED
                                                                → FAILED
```

Failed documents store `error_message`, `failed_stage`, and `retry_count`.

---

## Supported Document Types

| Format | Loader |
|---|---|
| PDF | PyMuPDF |
| TXT | chardet + UTF-8 |
| Markdown | Raw text (markdown-aware chunker) |
| HTML | BeautifulSoup4 |
| CSV | csv.DictReader → key:value rows |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/unit/          # fast, no services needed
pytest tests/integration/   # requires running Postgres + Qdrant
```

---

## Rebuild Qdrant Index

If Qdrant data is lost, rebuild from PostgreSQL:

```bash
python scripts/rebuild_qdrant_index.py --kb-id <kb_uuid>
```

---

## Environment Variables

See [`.env.example`](.env.example) for all variables with descriptions.

---

## Development Workflow

See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — local dev, migrations, PRs/CI, release and deploy.

## Production Deployment

- [`docs/DATABASE_SETUP.md`](docs/DATABASE_SETUP.md) — PostgreSQL + Qdrant production server setup (sizing, security, pooling, backups, HA).
- [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md) — full production-quality audit with a go-live checklist. **Read this before deploying** — it documents several issues (including a broken Alembic dependency and a file-upload path-traversal bug) that need fixing first.
