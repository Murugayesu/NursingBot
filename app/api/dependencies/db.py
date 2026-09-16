from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.postgres.database import get_session

# Re-export as a dependency alias for consistent imports across routes
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session

DBSession = Depends(get_db)
