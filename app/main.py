from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.storage.postgres.database import engine
from app.storage.qdrant.client import close_qdrant_client

# ── Structured logging setup ──────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logging.basicConfig(level=logging.INFO)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger = structlog.get_logger("app.startup")
    logger.info("starting rag platform", env=settings.app_env)
    yield
    # Cleanup
    logger.info("shutting down")
    await close_qdrant_client()
    await engine.dispose()


app = FastAPI(
    title="RAG Platform",
    description="Reusable, configuration-driven Retrieval-Augmented Generation platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
from app.api.routes.health import router as health_router
from app.api.routes.knowledge_bases import router as kb_router
from app.api.routes.ingestion_jobs import router as jobs_router
from app.api.routes.query import router as query_router
from app.api.routes.evaluation import router as eval_router

app.include_router(health_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(query_router)
app.include_router(eval_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "RAG Platform", "version": "1.0.0", "docs": "/docs"}
