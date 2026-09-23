from __future__ import annotations

import logging
import uuid as _uuid
from contextlib import asynccontextmanager

import structlog
import structlog.contextvars
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config.settings import get_settings
from app.storage.postgres.database import engine
from app.storage.qdrant.client import close_qdrant_client

# ── Structured logging setup ──────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
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


# ── #14: Enforce non-default secrets in production ────────────────────────────
def _validate_production_secrets() -> None:
    if settings.app_env != "production":
        return
    DEV_DEFAULTS = {"ragpassword", "password", "secret", "changeme", ""}
    if settings.postgres_password in DEV_DEFAULTS:
        raise RuntimeError(
            "POSTGRES_PASSWORD must be set to a strong value in APP_ENV=production"
        )
    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-placeholder"):
        raise RuntimeError(
            "OPENAI_API_KEY must be set in APP_ENV=production"
        )


# ── #18: Request ID middleware ────────────────────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request and bind it into structlog context."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── #9: Rate limiter (slowapi) ────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger = structlog.get_logger("app.startup")
    _validate_production_secrets()
    logger.info("starting rag platform", env=settings.app_env)
    yield
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

# ── #18: Request ID ───────────────────────────────────────────────────────────
app.add_middleware(RequestIDMiddleware)

# ── #9: Rate limiting ─────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── #6: CORS — explicit origins from env, not wildcard ───────────────────────
_allowed_origins = [
    o.strip()
    for o in getattr(settings, "cors_allowed_origins", "").split(",")
    if o.strip()
] or (["*"] if settings.app_env == "development" else [])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=bool(_allowed_origins and "*" not in _allowed_origins),
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# ── #18: Global exception handler ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger = structlog.get_logger("app.errors")
    logger.error(
        "unhandled exception",
        method=request.method,
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
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
