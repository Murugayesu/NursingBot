from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
    Runs RAGAS evaluation over an EvaluationDataset using the LangChain pipeline.

    For each question:
    1. Retrieve via HybridRetriever (LangChain BaseRetriever)
    2. Rerank via LocalJinaReranker (LangChain BaseDocumentCompressor)
    3. Generate via LCEL chain (ChatPromptTemplate → ChatOpenAI → StrOutputParser)
    4. Compute retrieval metrics (Recall@K, MRR, NDCG)
    5. Compute RAGAS generation metrics (Faithfulness, Answer Relevance, Correctness)
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

        from app.embeddings.lc_embeddings import get_embeddings
        from app.generation.chain import build_chain, format_docs_for_context
        from app.reranking.jina import LocalJinaReranker
        from app.retrieval.retriever import build_retriever

        run.status = EvaluationRunStatus.RUNNING
        await session.commit()

        embeddings = get_embeddings(
            model_name=rag_config.embedding.model,
            device=rag_config.embedding.device,
        )
        retriever = build_retriever(
            knowledge_base_id=knowledge_base_id,
            tenant_id=tenant_id,
            embeddings=embeddings,
            dense_top_k=rag_config.retrieval.dense_top_k,
            sparse_top_k=rag_config.retrieval.sparse_top_k,
            use_sparse=rag_config.retrieval.sparse,
        )
        reranker = LocalJinaReranker(
            model_name=rag_config.reranking.model,
            device=rag_config.reranking.device,
            top_n=rag_config.reranking.top_k,
        )
        chain = build_chain(rag_config)

        ragas_data: list[dict[str, Any]] = []
        per_question: list[dict[str, Any]] = []

        try:
            for q in questions:
                try:
                    # Retrieve
                    candidates = await retriever.ainvoke(q.question)

                    # Rerank
                    if rag_config.reranking.enabled and candidates:
                        reranked = await reranker.acompress_documents(candidates, q.question)
                    else:
                        reranked = candidates[: rag_config.reranking.top_k]

                    # Format context
                    context_str, _ = format_docs_for_context(reranked)

                    # Generate
                    answer = await chain.ainvoke(
                        {"context": context_str, "question": q.question}
                    )

                    retrieved_ids = [d.metadata.get("chunk_id", d.page_content[:50]) for d in candidates]
                    relevant_ids = [q.expected_context] if q.expected_context else []

                    pq = {
                        "question_id": str(q.id),
                        "question": q.question,
                        "answer": answer,
                        "contexts": [d.page_content for d in reranked],
                        "ground_truth": q.ground_truth,
                        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k=10),
                        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
                        "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_ids, k=10),
                    }
                    per_question.append(pq)
                    ragas_data.append({
                        "question": q.question,
                        "answer": answer,
                        "contexts": [d.page_content for d in reranked],
                        "ground_truth": q.ground_truth,
                    })
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
