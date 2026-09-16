from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config.rag_config import RAGConfig
from app.evaluation.metrics import (
    aggregate_retrieval_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from app.models.evaluation import EvaluationQuestion, EvaluationRun, EvaluationRunStatus

logger = structlog.get_logger(__name__)


class RAGASRunner:
    """
    Runs RAGAS evaluation over an EvaluationDataset.

    For each question:
    1. Run the full RAG pipeline (retrieval + generation)
    2. Compute retrieval metrics (Recall@K, MRR, NDCG)
    3. Compute RAGAS generation metrics (Faithfulness, Answer Relevance, Correctness)
    4. Store per-question results + aggregate scores in EvaluationRun
    """

    async def run(
        self,
        session: AsyncSession,
        run: EvaluationRun,
        questions: list[EvaluationQuestion],
        knowledge_base_id: uuid.UUID,
        tenant_id: str,
        rag_config: RAGConfig,
    ) -> EvaluationRun:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from app.embeddings import bge_m3
        from app.generation.context import ContextBuilder
        from app.generation.generator import LLMGenerator
        from app.reranking import jina
        from app.retrieval.hybrid import HybridRetriever

        run.status = EvaluationRunStatus.RUNNING
        await session.commit()

        retriever = HybridRetriever()
        ctx_builder = ContextBuilder(
            max_context_tokens=rag_config.generation.max_context_tokens
        )
        generator = LLMGenerator()

        ragas_data: list[dict[str, Any]] = []
        per_question: list[dict[str, Any]] = []

        try:
            for q in questions:
                try:
                    query_vec = await bge_m3.embed_query(
                        q.question,
                        model_name=rag_config.embedding.model,
                        device=rag_config.embedding.device,
                    )
                    candidates = await retriever.retrieve(
                        query=q.question,
                        query_vector=query_vec,
                        knowledge_base_id=knowledge_base_id,
                        tenant_id=tenant_id,
                        rag_config=rag_config,
                    )

                    if rag_config.reranking.enabled and candidates:
                        reranked = await jina.rerank(
                            query=q.question,
                            chunks=candidates,
                            top_k=rag_config.reranking.top_k,
                            model_name=rag_config.reranking.model,
                            device=rag_config.reranking.device,
                        )
                    else:
                        reranked = candidates[: rag_config.reranking.top_k]

                    context_str, sources = ctx_builder.build(reranked)
                    gen_result = await generator.generate(
                        query=q.question,
                        context_str=context_str,
                        sources=sources,
                        rag_config=rag_config,
                    )

                    retrieved_ids = [c.point_id for c in candidates]
                    relevant_ids = [q.expected_context] if q.expected_context else []

                    pq = {
                        "question_id": str(q.id),
                        "question": q.question,
                        "answer": gen_result.answer,
                        "contexts": [c.text for c in reranked],
                        "ground_truth": q.ground_truth,
                        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k=10),
                        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
                        "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_ids, k=10),
                    }
                    per_question.append(pq)
                    ragas_data.append(
                        {
                            "question": q.question,
                            "answer": gen_result.answer,
                            "contexts": [c.text for c in reranked],
                            "ground_truth": q.ground_truth,
                        }
                    )
                except Exception as qe:
                    logger.warning("evaluation question failed", error=str(qe), question_id=str(q.id))

            # ── RAGAS metrics ─────────────────────────────────────────────────
            ragas_scores: dict[str, float] = {}
            if ragas_data:
                dataset = Dataset.from_list(ragas_data)
                result = evaluate(
                    dataset,
                    metrics=[
                        context_precision,
                        context_recall,
                        faithfulness,
                        answer_relevancy,
                        answer_correctness,
                    ],
                )
                ragas_scores = dict(result)

            # ── Aggregate retrieval metrics ───────────────────────────────────
            agg = aggregate_retrieval_metrics(per_question, k=10)

            run.status = EvaluationRunStatus.COMPLETED
            run.context_precision = ragas_scores.get("context_precision")
            run.context_recall = ragas_scores.get("context_recall")
            run.faithfulness = ragas_scores.get("faithfulness")
            run.answer_relevance = ragas_scores.get("answer_relevancy")
            run.answer_correctness = ragas_scores.get("answer_correctness")
            run.recall_at_k = agg.get("recall_at_10")
            run.mrr = agg.get("mrr")
            run.ndcg = agg.get("ndcg_at_10")
            run.results = {
                "per_question": per_question,
                "aggregate_retrieval": agg,
                "ragas": ragas_scores,
            }

        except Exception as exc:
            logger.error("evaluation run failed", error=str(exc))
            run.status = EvaluationRunStatus.FAILED
            run.error_message = str(exc)

        await session.commit()
        return run
