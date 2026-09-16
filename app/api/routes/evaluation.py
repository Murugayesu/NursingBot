from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.db import get_db
from app.config.rag_config import RAGConfig
from app.evaluation.ragas_runner import RAGASRunner
from app.models.evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationRun,
    EvaluationRunStatus,
)
from app.models.knowledge_base import KnowledgeBase
from app.schemas.evaluation import (
    EvaluationDatasetCreate,
    EvaluationRunCreate,
    EvaluationRunResponse,
)

router = APIRouter(prefix="/knowledge-bases", tags=["evaluation"])


@router.post("/{kb_id}/datasets", status_code=201)
async def create_evaluation_dataset(
    kb_id: uuid.UUID,
    body: EvaluationDatasetCreate,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    dataset = EvaluationDataset(
        knowledge_base_id=kb_id,
        name=body.name,
        description=body.description,
        question_count=len(body.questions),
    )
    db.add(dataset)
    await db.flush()

    for q_data in body.questions:
        question = EvaluationQuestion(
            dataset_id=dataset.id,
            question=q_data["question"],
            ground_truth=q_data["ground_truth"],
            expected_context=q_data.get("expected_context"),
            extra_metadata={k: v for k, v in q_data.items()
                            if k not in {"question", "ground_truth", "expected_context"}},
        )
        db.add(question)

    await db.commit()
    await db.refresh(dataset)
    return {"id": dataset.id, "name": dataset.name, "question_count": dataset.question_count}


async def _run_evaluation(run_id: uuid.UUID, kb_id: uuid.UUID) -> None:
    from app.storage.postgres.database import AsyncSessionFactory

    async with AsyncSessionFactory() as session:
        run = await session.get(EvaluationRun, run_id)
        kb = await session.get(KnowledgeBase, kb_id)
        if not run or not kb:
            return

        questions_result = await session.execute(
            select(EvaluationQuestion).where(EvaluationQuestion.dataset_id == run.dataset_id)
        )
        questions = questions_result.scalars().all()

        rag_config = RAGConfig.from_dict(run.rag_config_snapshot or kb.rag_config)
        runner = RAGASRunner()
        await runner.run(
            session=session,
            run=run,
            questions=questions,
            knowledge_base_id=kb_id,
            tenant_id=kb.tenant_id,
            rag_config=rag_config,
        )


@router.post("/{kb_id}/evaluate", response_model=EvaluationRunResponse, status_code=202)
async def start_evaluation(
    kb_id: uuid.UUID,
    body: EvaluationRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    dataset = await db.get(EvaluationDataset, body.dataset_id)
    if not dataset or dataset.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")

    rag_config_snapshot = body.rag_config_override or kb.rag_config

    run = EvaluationRun(
        knowledge_base_id=kb_id,
        dataset_id=body.dataset_id,
        rag_config_snapshot=rag_config_snapshot,
        status=EvaluationRunStatus.PENDING,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_run_evaluation, run.id, kb_id)
    return run


@router.get("/{kb_id}/evaluation-runs/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation_run(
    kb_id: uuid.UUID,
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    run = await db.get(EvaluationRun, run_id)
    if not run or run.knowledge_base_id != kb_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run
