from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.storage.postgres.database import engine
from app.storage.qdrant.client import check_qdrant_health

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["health"])

_HEALTH_TIMEOUT = 2.0  # seconds — prevents hanging liveness probes (#12)


@router.get("/health", summary="Health check")
async def health():
    """
    Check connectivity to all backend services.
    Returns 200 if all dependencies are healthy, 503 if any are degraded.
    Each check has a hard 2s timeout so a hung connection never blocks a probe.
    """
    # ── Postgres ──────────────────────────────────────────────────────────────
    pg_ok = False
    try:
        async def _pg_check() -> None:
            import sqlalchemy
            async with engine.connect() as conn:
                await conn.execute(sqlalchemy.text("SELECT 1"))

        await asyncio.wait_for(_pg_check(), timeout=_HEALTH_TIMEOUT)
        pg_ok = True
    except asyncio.TimeoutError:
        logger.warning("postgres health check timed out")
    except Exception as exc:
        logger.warning("postgres health check failed", error=str(exc))

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_ok = False
    try:
        qdrant_ok = await asyncio.wait_for(
            check_qdrant_health(), timeout=_HEALTH_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.warning("qdrant health check timed out")
    except Exception as exc:
        logger.warning("qdrant health check failed", error=str(exc))

    all_ok = pg_ok and qdrant_ok
    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_ok else "degraded",
            "services": {
                "postgres": "ok" if pg_ok else "unreachable",
                "qdrant": "ok" if qdrant_ok else "unreachable",
            },
        },
    )
