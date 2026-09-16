from __future__ import annotations

from fastapi import APIRouter

from app.storage.postgres.database import engine
from app.storage.qdrant.client import check_qdrant_health

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check")
async def health():
    """Check connectivity to all backend services."""
    # Postgres
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        pg_ok = True
    except Exception as e:
        pg_ok = False

    # Qdrant
    qdrant_ok = await check_qdrant_health()

    all_ok = pg_ok and qdrant_ok
    return {
        "status": "ok" if all_ok else "degraded",
        "services": {
            "postgres": "ok" if pg_ok else "unreachable",
            "qdrant": "ok" if qdrant_ok else "unreachable",
        },
    }
