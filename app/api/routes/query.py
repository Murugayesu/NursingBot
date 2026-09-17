from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.db import get_db
from app.config.rag_config import RAGConfig
from app.embeddings.lc_embeddings import get_embeddings
from app.generation.chain import build_chain, format_docs_for_context
from app.models.knowledge_base import KnowledgeBase
from app.observability.langfuse_tracer import get_langfuse_handler
from app.reranking.jina import LocalJinaReranker
from app.retrieval.retriever import build_retriever
from app.schemas.query import QueryRequest, QueryResponse, SourceResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/knowledge-bases", tags=["query"])


@router.post("/{kb_id}/query", response_model=QueryResponse)
async def query_knowledge_base(
    kb_id: uuid.UUID,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    rag_config = RAGConfig.from_dict(kb.rag_config)
    log = logger.bind(kb_id=str(kb_id), query=body.query[:80])
    log.info("query received")

    # ── 1. Build embeddings + retriever ──────────────────────────────────────
    embeddings = get_embeddings(
        model_name=rag_config.embedding.model,
        device=rag_config.embedding.device,
    )
    retriever = build_retriever(
        knowledge_base_id=kb_id,
        tenant_id=kb.tenant_id,
        embeddings=embeddings,
        dense_top_k=rag_config.retrieval.dense_top_k,
        sparse_top_k=rag_config.retrieval.sparse_top_k,
        use_sparse=rag_config.retrieval.sparse,
        extra_filters=body.filters or None,
    )

    # ── 2. Retrieve candidates ────────────────────────────────────────────────
    log.info("retrieving")
    candidates = await retriever.ainvoke(body.query)
    log.info("retrieved", candidates=len(candidates))

    # ── 3. Rerank with local Jina ─────────────────────────────────────────────
    if rag_config.reranking.enabled and candidates:
        reranker = LocalJinaReranker(
            model_name=rag_config.reranking.model,
            device=rag_config.reranking.device,
            top_n=body.top_k or rag_config.reranking.top_k,
        )
        reranked = await reranker.acompress_documents(candidates, body.query)
        log.info("reranked", kept=len(reranked))
    else:
        reranked = candidates[: (body.top_k or rag_config.reranking.top_k)]

    # ── 4. Format context + extract sources ──────────────────────────────────
    context_str, sources = format_docs_for_context(reranked)

    # ── 5. Run LCEL chain with Langfuse tracing ───────────────────────────────
    handler = get_langfuse_handler(
        trace_name="rag_query",
        metadata={"kb_id": str(kb_id), "query": body.query},
    )
    callbacks = [handler] if handler else []

    chain = build_chain(rag_config)
    log.info("generating answer")
    answer = await chain.ainvoke(
        {"context": context_str, "question": body.query},
        config={"callbacks": callbacks},
    )

    # ── 6. Token usage (best-effort from chain metadata) ─────────────────────
    # ChatOpenAI reports usage via the callback; we return 0 if unavailable
    return QueryResponse(
        query=body.query,
        answer=answer,
        sources=[SourceResponse(**s) for s in sources] if body.include_sources else [],
        prompt_version=rag_config.prompt.version,
        model=rag_config.generation.model,
        input_tokens=0,
        output_tokens=0,
    )
