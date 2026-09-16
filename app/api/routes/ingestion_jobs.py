from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.db import get_db
from app.models.ingestion_job import IngestionJob
from app.schemas.document import IngestionJobResponse

router = APIRouter(prefix="/ingestion-jobs", tags=["ingestion"])


@router.get("/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(IngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job
